"""无依赖静态自检（不需要 torch / flask / 网络）。

用途：在任何有 Python 3.9+ 的机器上先跑这个，
      快速暴露语法错误、导入错误、被禁止的匹配逻辑、以及路径安全问题。
      全部通过后再去装重依赖跑 pytest。

用法：
    python selfcheck.py
"""
from __future__ import annotations

import ast
import builtins as _builtins_mod
import csv
import io
import re
import sys
import tokenize
import traceback
from pathlib import Path

# ---------------------------------------------------------------- 编码兜底
# Windows 控制台默认可能是 GBK(936)。本脚本会输出中文，
# 此时 print() 会抛 UnicodeEncodeError，整个自检第一步就崩。
# 这里在**任何中文输出之前**把标准输出切到 UTF-8 并放宽错误处理。
def _force_utf8_stdout() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # 流已被重定向或不可重配置时忽略，不影响功能
            pass


_force_utf8_stdout()

BASE = Path(__file__).resolve().parent
FAILURES: list[str] = []
PASSES: list[str] = []


def ok(name: str) -> None:
    PASSES.append(name)
    print(f"  [PASS] {name}")


def fail(name: str, detail: str) -> None:
    FAILURES.append(f"{name}: {detail}")
    print(f"  [FAIL] {name}: {detail}")


# ------------------------------------------------------------------ 语法
def check_syntax() -> None:
    print("\n[1] 语法检查（AST 解析）")
    files = sorted(BASE.glob("*.py")) + sorted((BASE / "tests").glob("*.py"))
    for path in files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            ok(f"{path.name} 语法正确")
        except SyntaxError as exc:
            fail(path.name, f"第 {exc.lineno} 行语法错误：{exc.msg}")
        except (OSError, UnicodeDecodeError) as exc:
            # 文件读不出来时不能让异常逃逸，否则整个自检中断，
            # 后面的检查一项都跑不到，反而看不出真正的问题。
            fail(path.name, f"无法读取或解码：{exc}")


# -------------------------------------------------- 未定义名称（近似）
# 已多次出现"用了没导入的名字"（re、UnidentifiedImageError），
# 语法检查发现不了，只有运行到那一行才炸。这里做静态近似检测。
# 注意：不要用 dir(__builtins__)。它在 __main__ 里是模块、在被导入时是 dict，
# 两条分支结果完全不同，写错会让内置函数全部被判为"未定义"而大面积误报。
_BUILTINS = set(dir(_builtins_mod)) | {
    "__file__", "__name__", "__doc__", "__spec__",
    "__loader__", "__package__", "__debug__", "self", "cls",
}


def _collect_bound_names(tree: ast.AST) -> set[str]:
    """收集模块内被绑定的名字：导入、赋值、函数/类定义、参数、for、with、except。"""
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
        elif isinstance(node, ast.alias):
            bound.add((node.asname or node.name).split(".")[0])
    return bound


def check_undefined_names() -> None:
    print("\n[13] 未定义名称检查（捕获漏导入）")
    files = sorted(BASE.glob("*.py")) + sorted((BASE / "tests").glob("*.py"))
    problems: list[str] = []
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, OSError, UnicodeDecodeError):
            continue      # 由语法检查负责报告
        bound = _collect_bound_names(tree) | _BUILTINS
        for node in ast.walk(tree):
            # 只看"被读取"的裸名字；属性访问（x.y）与关键字实参不算
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id not in bound:
                    problems.append(f"{path.name}:{node.lineno} 未定义 {node.id}")
    if problems:
        shown = problems[:12]
        fail("未定义名称",
             f"疑似漏导入（共 {len(problems)} 处）：" + "；".join(shown))
    else:
        ok(f"{len(files)} 个文件均未发现漏导入的裸名字")


# -------------------------------------------------- 被禁止的匹配逻辑
def strip_comments_and_strings(source: str) -> str:
    """去掉注释与字符串字面量，只保留可执行代码。

    这一步很关键：matcher.py 的模块文档里会**列举**被禁止的术语
    （如 top2_mean、threshold）来说明"我们不这么做"。若直接全文搜索，
    这些说明文字会被误判成违规代码。

    失败时**不**静默回退到原始源码——那会让文档里的术语变成假失败，
    排查方向完全被带偏。这里直接抛出，由调用方报告为检查器自身的错误。
    """
    pieces: list[str] = []
    reader = io.StringIO(source).readline
    try:
        for token in tokenize.generate_tokens(reader):
            if token.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            pieces.append(token.string)
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        raise RuntimeError(f"无法分词，禁止逻辑检查不可信：{exc}") from exc
    return " ".join(pieces)


def _function_source(source: str, func_name: str) -> str | None:
    """返回源码中某个顶层函数的完整定义文本；找不到则返回 None。

    用具名函数的 AST 边界来截取，而不是 source.find(...) 加一个定长偏移。
    定长偏移（例如 start+1200）会随着注释或代码增多而悄悄截断：
    被检查的语句一旦落到窗口之外，就会报出"缺少某校验"的**假失败**，
    而代码其实是对的。这里按 end_lineno 取到函数结尾，不受长度影响。
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    lines = source.splitlines(keepends=True)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            if node.end_lineno is None:
                return None
            return "".join(lines[node.lineno - 1:node.end_lineno])
    return None


def check_prohibited_logic() -> None:
    print("\n[2] 禁止逻辑检查（仅扫描可执行代码，跳过注释与字符串）")
    matcher = BASE / "matcher.py"
    code = strip_comments_and_strings(matcher.read_text(encoding="utf-8"))

    # 禁止术语只在此处定义一次，避免两处清单各自漂移
    PROHIBITED = (
        ("top2", "禁止 top2 平均策略"),
        ("threshold", "禁止最低匹配阈值"),
        ("MIN_SIMILARITY", "禁止相似度下限"),
        ("confidence", "禁止置信度分类"),
        ("argsort", "禁止排序取 TopN（只允许 argmax）"),
        ("topk", "禁止 TopK"),
    )

    for forbidden, why in PROHIBITED:
        if forbidden in code:
            fail(f"matcher 不含 {forbidden}", why)
        else:
            ok(f"matcher 不含 {forbidden}")

    if "argmax" in code:
        ok("matcher 使用 argmax 取最高分")
    else:
        fail("matcher 使用 argmax", "未找到 argmax")

    # 上述术语若被搬到别的模块（例如在路由层加阈值、在预处理阶段做歌曲聚合），
    # 只扫 matcher.py 就查不出来。因此对**全部生产模块**再扫一遍。
    # 不扫 selfcheck.py 与 tests/：前者必然包含这些字面量，后者靠自身用例覆盖。
    offenders: list[str] = []
    for path in sorted(BASE.glob("*.py")):
        if path.name in ("selfcheck.py", "matcher.py"):
            continue
        try:
            other = strip_comments_and_strings(path.read_text(encoding="utf-8"))
        except RuntimeError as exc:
            fail(f"{path.name} 可分词", str(exc))
            continue
        for forbidden, _why in PROHIBITED:
            if forbidden in other:
                offenders.append(f"{path.name}: {forbidden}")

    if offenders:
        fail("全项目无禁止术语", f"以下模块出现被禁止的术语：{offenders}")
    else:
        ok("其余生产模块均未出现阈值 / top2 / 置信度 / 排序取 TopN 等术语")

    # MatchResult 不得有分数字段
    text = matcher.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "MatchResult":
            fields = set()
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    fields.add(stmt.target.id)
            expected = {"song_id", "matched_image_id", "degraded", "degrade_reason"}
            if fields == expected:
                ok(f"MatchResult 字段恰为 {sorted(expected)}（无 score/similarity）")
            else:
                fail("MatchResult 字段", f"实际为 {sorted(fields)}，期望 {sorted(expected)}")
            break
    else:
        fail("MatchResult 存在", "未找到该类")


# ------------------------------------------------------------ 离线防线
def check_offline_guards() -> None:
    print("\n[3] 离线防线检查（clip_encoder.py）")
    path = BASE / "clip_encoder.py"
    source = path.read_text(encoding="utf-8")
    # 关键：只看可执行代码，避免文档字符串里的字样造成假通过
    code = strip_comments_and_strings(source)

    # 环境变量名天然以**字符串字面量**形式出现
    # （os.environ.setdefault("HF_HUB_OFFLINE", "1")），
    # 而 code 已剥掉所有字符串，因此这里必须查 source，否则必然假失败。
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if re.search(rf'["\']{var}["\']', source):
            ok(f"设置 {var}")
        else:
            fail(f"设置 {var}", "未找到对应的环境变量设置")

    # local_files_only=True 是关键字实参，名字保留在 code 中
    if "local_files_only" in code:
        ok("from_pretrained 使用 local_files_only=True")
    else:
        fail("local_files_only", "未在可执行代码中找到")

    if "inference_mode" in code:
        ok("使用 torch.inference_mode()")
    else:
        fail("inference_mode", "未找到（计划书点名要求，不得用 no_grad 代替）")

    # 要求真实的 .eval() 调用，而不是匹配到 evaluate / retrieval 之类的词。
    #
    # code 是 " ".join(token.string) 拼出来的，model.eval() 会变成
    # "model . eval ( )"——点号与 eval 之间有空格。所以不能要求 ".eval"
    # 紧挨着写，必须允许点号两侧出现空白，否则永远匹配不到。
    if re.search(r"\.\s*eval\s*\(\s*\)", code):
        ok("模型显式调用 .eval()")
    else:
        fail(".eval()", "未找到形如 xxx.eval() 的调用")

    # 环境变量必须在 import torch / transformers 之前设置（按源码位置判断）。
    #
    # 两侧都锚定**代码形态**，且都取自同一份 code，偏移量才可比。
    #
    # 不能依赖 "HF_HUB_OFFLINE" 字面量：它在 code 中被剥离，
    # 直接 find 必然返回 -1（这正是上一版恒判失败的原因）。
    # 也不能改查 source：clip_encoder.py 的模块 docstring 里写着
    # "离线环境变量必须在 import transformers 之前设置"，
    # 在保留字符串的文本里 find("import transformers") 会命中该说明文字，
    # 位置比第 18 行的真实设置更靠前，判定同样失败。
    # 而 code 已剥掉字符串，那句话只剩"之前设置"，不可能再被命中。
    idx_env = code.find("environ . setdefault")
    if idx_env < 0:
        idx_env = code.find("os.environ.setdefault")
    idx_import = code.find("import torch")
    # transformers 在本项目里是 `from transformers import ...` 的形式，
    # 并不包含子串 "import transformers"（词序不同）。
    # 两种写法都要认，否则漏判一处导入点。
    idx_tf = code.find("import transformers")
    if idx_tf < 0:
        idx_tf = code.find("from transformers import")
    candidates = [x for x in (idx_import, idx_tf) if x > 0]
    if idx_env > 0 and candidates and idx_env < min(candidates):
        ok("离线环境变量在 import torch/transformers 之前设置")
    else:
        fail("环境变量顺序", "必须在 import transformers 之前设置，否则不生效")

    # 不得使用 no_grad 代替 inference_mode
    if "no_grad" in code:
        fail("未使用 no_grad", "计划书要求用 inference_mode")
    else:
        ok("未使用 no_grad（符合计划书要求）")

    # ---- Web 阶段的**结构性**离线保证（比"不调用"更强）----
    # 计划书 §3.3：Web 演示阶段禁止联网；§3.3 末条：预处理不得在 Flask 请求中执行。
    # 只要 Web 的 import 图里根本没有 requests/urllib/socket，也没有 preprocess，
    # 那么"忘了断网"这件事在结构上就不可能发生。
    # 名单不再手工维护：从 app.py 出发，沿**项目内**的 import 关系求传递闭包。
    # 手工列名单会漂移——例如 clip_encoder 是在 create_app 内部按需导入的
    # （app.py 里 `from clip_encoder import load_encoder_or_none`），
    # 手工名单很容易漏掉它，而它恰恰是唯一碰 torch/transformers 的模块。
    local_names = {p.name for p in BASE.glob("*.py")}
    web_modules: set[str] = set()
    pending = ["app.py"]
    while pending:
        name = pending.pop()
        if name in web_modules or name not in local_names:
            continue
        web_modules.add(name)
        for line in (BASE / name).read_text(
                encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if stripped.startswith("import "):
                candidates = [m.strip().split(" as ")[0].strip()
                              for m in stripped[len("import "):].split(",")]
            elif stripped.startswith("from "):
                candidates = [stripped[len("from "):].split(" import ")[0].strip()]
            else:
                continue
            for candidate in candidates:
                target = candidate.split(".")[0] + ".py"
                if target in local_names:
                    pending.append(target)

    # 只放**顶层模块名**：比较时取 `from urllib.request import x` 的根名 urllib。
    network_names = {"requests", "urllib", "urllib3", "http", "socket",
                     "ftplib", "httpx", "aiohttp"}
    offenders: list[str] = []
    for name in sorted(web_modules):
        path = BASE / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if not (stripped.startswith("import ") or stripped.startswith("from ")):
                continue
            # 逐个解析被导入的模块名，而不是对整行做前缀匹配——
            # 否则 `import os, requests` 这类多模块写法会漏掉，
            # `from urllib.request import urlopen` 也容易判错。
            if stripped.startswith("import "):
                modules = [m.strip().split(" as ")[0].strip()
                           for m in stripped[len("import "):].split(",")]
            else:  # from X import ...
                modules = [stripped[len("from "):].split(" import ")[0].strip()]
            for module in modules:
                root = module.split(".")[0]
                if root in network_names:
                    offenders.append(f"{name}: {stripped}")

    if offenders:
        fail("Web 阶段结构性离线",
             f"Web import 图里出现了联网模块：{offenders}")
    else:
        ok(f"Web 的 {len(web_modules)} 个模块均未导入联网库（结构性离线）")

    # 预处理必须是独立批处理程序，不得被 Flask 导入
    app_src = (BASE / "app.py").read_text(encoding="utf-8") \
        if (BASE / "app.py").exists() else ""
    if re.search(r"^\s*(import\s+preprocess|from\s+preprocess\s+import)", app_src,
                 re.MULTILINE):
        fail("预处理隔离", "app.py 导入了 preprocess（计划书 §3.3 禁止）")
    else:
        ok("app.py 未导入 preprocess（预处理保持独立批处理）")


# ------------------------------------------------------------ 路径安全
def check_path_safety() -> None:
    print("\n[4] 路径安全逻辑检查（不依赖第三方库）")

    # 直接从源码里提取判定函数的行为约定，做纯字符串级验证
    source = (BASE / "image_utils.py").read_text(encoding="utf-8")

    if "startswith(config.STATIC_PATH_PREFIXES)" in source:
        ok("强制静态路径前缀白名单")
    else:
        fail("路径前缀白名单", "未找到")

    if '".."' in source:
        ok("拒绝父目录穿越 ..")
    else:
        fail("拒绝 ..", "未找到")

    if "relative_to" in source:
        ok("使用 relative_to 做目录归属校验")
    else:
        fail("目录归属校验", "未找到 relative_to")

    # upload_path_is_safe / audio_path_is_safe 必须存在
    for fn in ("upload_path_is_safe", "audio_path_is_safe",
               "static_relative_to_absolute", "normalize_once"):
        if f"def {fn}" in source:
            ok(f"存在 {fn}()")
        else:
            fail(f"{fn}()", "未定义")

    # save_comment_image 用 song_key 直接拼路径，必须自行校验，
    # 不能只依赖调用方——否则任何一次直接调用都可能写出目录之外。
    # 按**函数体范围**取文本，而不是 source[start:start+1200] 这种定长切片：
    # 定长切片会随注释增多而悄悄截断，把本来存在的校验切到窗口之外，
    # 于是报出一条"缺少校验"的假失败。
    body = _function_source(source, "save_comment_image")
    if body is None:
        # 不能静默跳过：函数被改名或文件无法解析时，
        # 若什么都不输出，报告里看不到任何异常，而这两项安全检查其实没跑。
        fail("save_comment_image 可定位",
             "无法在 image_utils.py 中定位该函数（被改名？文件有语法错误？）")
    else:
        if "is_valid_song_key" in body:
            ok("save_comment_image 内部自行校验 song_key")
        else:
            fail("save_comment_image 校验",
                 "直接使用 song_key 拼路径却未校验，安全依赖调用方自觉")
        if "is_within" in body:
            ok("save_comment_image 二次确认路径未越界（纵深防御）")
        else:
            fail("save_comment_image 越界确认", "缺少最终路径归属校验")

    # 幂等入口必须也强制体积上限，否则"已规范化"捷径会绕过限制
    fn_body = _function_source(source, "normalize_once")
    if fn_body is None:
        fail("normalize_once 可定位",
             "无法在 image_utils.py 中定位该函数（被改名？文件有语法错误？）")
    elif "len(data) > limit" in fn_body:
        ok("normalize_once 在幂等分支之前强制体积上限")
    else:
        fail("normalize_once 体积上限",
             "未在幂等判断前检查大小，可能被用于绕过上传限制")


# ------------------------------------------------------- 数据库约束
def check_db_constraints() -> None:
    print("\n[5] 数据库约束检查（db.py）")
    source = (BASE / "db.py").read_text(encoding="utf-8")

    if "PRAGMA foreign_keys = ON" in source:
        ok("每个连接开启 PRAGMA foreign_keys")
    else:
        fail("外键约束", "未找到 PRAGMA foreign_keys = ON")

    for table in ("songs", "song_comment_images", "posts"):
        if f"CREATE TABLE IF NOT EXISTS {table}" in source:
            ok(f"建表 {table}")
        else:
            fail(f"建表 {table}", "未找到")

    if "UNIQUE(song_id, image_sha256)" in source:
        ok("评论图片按 (song_id, image_sha256) 去重")
    else:
        fail("去重唯一约束", "未找到")

    # 事务语义：回滚必须覆盖所有异常，而非只捕获 sqlite3.Error
    if "except BaseException:" in source and "rollback" in source:
        ok("事务回滚覆盖所有异常类型（含 KeyboardInterrupt）")
    else:
        fail("事务回滚范围",
             "get_connection 只捕获 sqlite3.Error，非 SQL 异常逃逸时不会回滚")

    # 连接失败必须同时覆盖 sqlite3.Error 与 OSError
    if "except (sqlite3.Error, OSError)" in source:
        ok("连接失败归一化覆盖 sqlite3.Error 与 OSError")
    else:
        fail("连接失败归一化",
             "mkdir 等 OSError 未被捕获，DatabaseUnavailable 不会抛出，503 不生效")

    # 检查是否有 f-string 拼接用户输入的 SQL（参数化检查）
    bad = []
    for line_no, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if ("execute(" in line or "executemany(" in line) and "%" in line and "?" not in line:
            bad.append(str(line_no))
    if bad:
        fail("SQL 参数化", f"疑似字符串拼接 SQL 的行：{','.join(bad)}")
    else:
        ok("SQL 使用参数化占位符")


# ------------------------------------------------------ 兜底边界
def check_fallback_boundary() -> None:
    print("\n[6] 兜底边界检查（matcher.py + app.py）")
    matcher = (BASE / "matcher.py").read_text(encoding="utf-8")
    app_src = (BASE / "app.py").read_text(encoding="utf-8")

    # matcher 中 fallback 只应在四种情况触发
    for marker in ("模型不可用", "没有有效的评论图片向量", "图片推理失败"):
        if marker in matcher:
            ok(f"兜底分支存在：{marker}")
        else:
            fail(f"兜底分支 {marker}", "未找到")

    # app 中图片校验失败不得走兜底
    if "ImageValidationError" in app_src and "flash(str(exc)" in app_src:
        ok("图片校验失败走友好报错，不触发兜底")
    else:
        fail("图片校验失败处理", "未找到友好报错分支")

    # 降级时页面必须明确提示
    result_tpl = (BASE / "templates" / "result.html").read_text(encoding="utf-8")
    if "星星暂时被云遮住了" in result_tpl:
        ok("降级页面用故事化文案提示默认歌曲")
    else:
        fail("降级提示", "result.html 缺少说明文字")

    # 降级时不展示评论图
    if "{% if matched_image_path %}" in result_tpl:
        ok("降级时（无匹配图）不展示评论图片")
    else:
        fail("降级图片展示", "未找到条件渲染")


# ------------------------------------------------------ 模板完整性
def check_templates() -> None:
    print("\n[7] 模板与静态资源检查")
    tpl_dir = BASE / "templates"
    expected = ["base.html", "index.html", "result.html",
                "song_detail.html", "posts.html", "error.html"]
    for name in expected:
        path = tpl_dir / name
        if path.exists():
            ok(f"模板存在 {name}")
        else:
            fail(f"模板 {name}", "缺失")

    css = BASE / "static" / "css" / "style.css"
    if css.exists() and css.stat().st_size > 0:
        ok("本地样式表存在且非空")
    else:
        fail("样式表", "缺失或为空")

    # 不得引用 CDN（允许 XML 命名空间声明，如 xmlns="http://www.w3.org/1999/xhtml"）
    EXTERNAL = re.compile(r"""(?:src|href)\s*=\s*["']https?://""", re.IGNORECASE)
    for path in sorted(tpl_dir.glob("*.html")):
        text = path.read_text(encoding="utf-8")
        if EXTERNAL.search(text):
            fail(f"{path.name} 引用外部资源", "计划书要求离线，不得加载 CDN")
        else:
            ok(f"{path.name} 无外部资源引用")

    # CSRF token 出现在所有 POST 表单中
    for name in ("index.html", "result.html"):
        text = (tpl_dir / name).read_text(encoding="utf-8")
        if "csrf_token" in text:
            ok(f"{name} 含 CSRF token")
        else:
            fail(f"{name} CSRF", "POST 表单缺少 csrf_token")

    # 模板里 url_for 引用的 endpoint 必须真实存在，否则页面一渲染就 500
    app_src = (BASE / "app.py").read_text(encoding="utf-8")
    defined = set(re.findall(r"^    def ([a-z_]+)\(", app_src, re.MULTILINE))
    defined.add("static")  # Flask 内置

    referenced: set[str] = set()
    for path in sorted(tpl_dir.glob("*.html")):
        text = path.read_text(encoding="utf-8")
        # 单双引号都要认：只匹配单引号时，将来若有人写成
        # url_for("index")，该引用会被**静默漏掉**，检查形同虚设。
        referenced |= set(re.findall(r"""url_for\(\s*['"]([a-z_]+)['"]""", text))

    unknown = sorted(referenced - defined)
    if unknown:
        fail("模板 endpoint 有效", f"以下 url_for 目标在 app.py 中不存在：{unknown}")
    else:
        ok(f"模板引用的 {len(referenced)} 个 endpoint 均已定义")

    # 不得使用 |safe 绕过转义（留言内容为用户输入）
    unescaped = [
        path.name for path in sorted(tpl_dir.glob("*.html"))
        if "|safe" in path.read_text(encoding="utf-8")
    ]
    if unescaped:
        fail("模板未绕过转义", f"以下模板使用了 |safe：{unescaped}")
    else:
        ok("模板未使用 |safe，用户内容默认转义")

    # ---- 本地样式（计划书 §3.3：不得加载 CDN 字体/脚本/样式）----
    # 页面样式现在拆成三份：基础样式、小王子主题、加载层。
    # 类覆盖检查必须看全部本地 CSS，否则会把主题/加载层类误报为缺失。
    css_paths = sorted((BASE / "static" / "css").glob("*.css"))
    style_path = BASE / "static" / "css" / "style.css"
    if not style_path.exists():
        fail("本地样式", "static/css/style.css 缺失")
    elif not css_paths:
        fail("本地样式", "static/css 下没有任何 CSS 文件")
    else:
        css_parts = []
        for path in css_paths:
            css_parts.append(path.read_text(encoding="utf-8"))
        css = "\n".join(css_parts)
        # data: URI 里的 xmlns 会包含 http://www.w3.org/...，它不是网络请求。
        css_for_remote = re.sub(r"url\(\s*['\"]?data:[^)]*\)", "", css)
        if len(style_path.read_text(encoding="utf-8").strip()) < 500:
            fail("本地样式", "style.css 疑似占位文件")
        elif re.search(r"@import|url\(\s*['\"]?https?:|//[a-z0-9.-]+\.[a-z]{2,}/", css_for_remote):
            fail("样式无远程依赖", "本地 CSS 引用了外部资源（违反 §3.3 离线要求）")
        else:
            ok(f"本地样式 {len(css)} 字节，无 CDN / @import 远程引用"
               f"（{len(css_paths)} 个文件）")

        # 模板用到的 class 至少要有定义，否则页面样式大面积失效
        used: set[str] = set()
        for path in sorted(tpl_dir.glob("*.html")):
            for attr in re.findall(r'class="([^"]*)"', path.read_text(encoding="utf-8")):
                # 含 Jinja 插值的整个 class 属性直接跳过。
                # 不能只按空格拆分后逐个判断：'flash-{{ category }}'
                # 会被拆成 'flash-{{' / 'category' / '}}'，
                # 其中 'category' 是不带花括号的裸名字，会漏网并被当成真实类名。
                if "{" in attr or "}" in attr:
                    continue
                for name in attr.split():
                    used.add(name)
        missing_css = sorted(c for c in used if f".{c}" not in css)
        if missing_css:
            fail("样式类覆盖", f"以下 class 在本地 CSS 中没有定义：{missing_css}")
        else:
            ok(f"模板使用的 {len(used)} 个 class 均已在本地 CSS 中定义")

        # flash-{{ category }} 是动态拼出来的，上面的循环跳过它，
        # 因此单独校验 app.py 里实际用到的每个 category 都有对应样式。
        categories = set(re.findall(r'flash\([^)]*,\s*"([a-z_]+)"\s*\)', app_src))
        missing_flash = sorted(c for c in categories if f".flash-{c}" not in css)
        if missing_flash:
            fail("提示样式覆盖",
                 f"flash 分类 {missing_flash} 在本地 CSS 中没有 .flash-<分类> 样式")
        else:
            ok(f"app.py 用到的 {len(categories)} 个 flash 分类均有样式"
               f"（{', '.join(sorted(categories))}）")


# ------------------------------------------------------ 交付物清单
def check_deliverables() -> None:
    print("\n[8] 交付物清单检查（计划书 §18）")
    required = [
        "app.py", "config.py", "db.py", "clip_encoder.py",
        "matcher.py", "image_utils.py", "preprocess.py",
        "hybrid.py", "text_encoder.py", "build_index.py",
        "selfcheck.py", "verify_discrimination.py", "pytest.ini",
        "run_selfcheck.cmd", "run_selfcheck_ascii.cmd",
        "stage_assets_ascii.cmd",
        "run_selfcheck.ps1", "stage_assets.ps1", "setup_env.ps1",
        "requirements.txt", "README.md", "VERIFICATION.md", "LOADER_STYLE.md", ".gitignore",
        "data/songs_seed.csv", "data/comment_images_manifest.csv",
        "templates/access_login.html", "templates/base.html", "templates/index.html", "templates/result.html",
        "templates/song_detail.html", "templates/posts.html", "templates/error.html",
        "templates/postcard.html", "templates/_little_prince_loader.html",
        "static/css/style.css", "static/css/little-prince-theme.css",
        "static/css/little-prince-loader.css",
        "static/js/little-prince-loader.js", "static/js/site-loader.js",
        "static/js/song.js", "static/js/audio-continuity.js",
        "tests/__init__.py", "tests/conftest.py",
        "tests/test_image_validation.py", "tests/test_matcher.py",
        "tests/test_preprocess.py", "tests/test_routes.py",
        "tests/test_db_transactions.py", "tests/test_hybrid.py", "tests/test_wall_search.py",
        "tests/test_access_gate.py",
        "tests/test_text_calibration.py",
    ]
    for rel in required:
        path = BASE / rel
        if path.exists():
            ok(f"存在 {rel}")
        else:
            fail(f"交付物 {rel}", "缺失")

    # 反向检查：tests/ 下不能有未被清单登记的测试文件（防止漏登记）
    tests_dir = BASE / "tests"
    if tests_dir.exists():
        unlisted = sorted(
            p.name for p in tests_dir.glob("test_*.py")
            if f"tests/{p.name}" not in required
        )
        if unlisted:
            fail("测试清单完整", f"tests/ 下存在未登记的文件：{unlisted}")
        else:
            ok("tests/ 下全部测试文件均已登记")

    # 反向检查：顶层也不该出现未登记的 .py。
    # 只做正向检查时，新增模块不会报错，却可能悄悄绕过
    # 离线检查、禁止术语检查等按名单遍历的检查——正是上一版
    # 漏掉 clip_encoder.py 的那类问题。
    unlisted_modules = sorted(
        p.name for p in BASE.glob("*.py") if p.name not in required
    )
    if unlisted_modules:
        fail("顶层模块清单完整",
             f"以下顶层模块未登记进 required（需同步确认它是否该被其他检查覆盖）："
             f"{unlisted_modules}")
    else:
        ok(f"顶层 {len(list(BASE.glob('*.py')))} 个模块均已登记")

    # 不得有空文件冒充音频
    audio_dir = BASE / "static" / "audio"
    if audio_dir.exists():
        audio_files = [p for p in audio_dir.iterdir() if p.suffix.lower() in
                       (".mp3", ".wav", ".m4a", ".ogg", ".flac")]
        if audio_files:
            empties = [p.name for p in audio_files if p.stat().st_size == 0]
            if empties:
                fail("音频非空", f"发现空文件冒充音频：{empties}")
            else:
                ok(f"音频目录有 {len(audio_files)} 个非空文件")
        else:
            ok("音频目录为空（素材待提供，未伪造）")


def check_test_isolation() -> None:
    """测试必须隔离所有会写盘的路径，否则会污染真实项目目录。"""
    print("\n[9] 测试隔离检查")
    path = BASE / "tests" / "test_routes.py"
    if not path.exists():
        fail("test_routes.py", "缺失")
        return

    source = path.read_text(encoding="utf-8")

    # 哪些常量会写盘？直接问 config.py，不要手工列名单——
    # 手工名单会与 ensure_directories() 漂移：新增一个目录后，
    # 测试仍会在真实项目里创建它，而且是静默的。
    config_src = (BASE / "config.py").read_text(encoding="utf-8")
    block = re.search(r"def ensure_directories\(.*?\n    for directory in \((.*?)\):",
                      config_src, re.DOTALL)
    if not block:
        fail("解析 ensure_directories", "无法从 config.py 提取目录常量列表")
        return
    created = re.findall(r"[A-Z_][A-Z0-9_]*", block.group(1))

    # 另外两个是**文件**（不是目录），但同样会被写入真实项目，必须一起隔离。
    file_targets = ["DB_PATH", "SECRET_KEY_FILE"]
    writable = created + file_targets

    missing = [name for name in writable if f'setattr(config, "{name}"' not in source]
    if missing:
        fail("测试路径隔离",
             f"以下会写盘的常量未被 monkeypatch，pytest 会污染真实目录：{missing}")
    else:
        ok(f"测试隔离覆盖 config.ensure_directories() 的 {len(created)} 个目录"
           f" + {len(file_targets)} 个文件目标（共 {len(writable)} 项）")

    # fixture 必须走统一的隔离函数，避免两处设置漂移
    if source.count("_isolate_config(") >= 3:  # 1 处定义 + 至少 2 处调用
        ok("隔离逻辑集中在一个函数中复用")
    else:
        fail("隔离逻辑复用",
             "存在重复的 monkeypatch 代码块，容易漏改其中一处")

    if "MODEL_DOWNGRADE" in source:
        ok("测试使用降级模式，不依赖真实 CLIP 模型")
    else:
        fail("测试降级模式", "测试不应依赖真实模型")


def check_docs_consistency() -> None:
    """文档不得与实际文件脱节。

    VERIFICATION.md 是人工照做的操作手册：步骤编号重复、测试表格漏列
    文件，都会让人照着做完却以为覆盖到了。
    """
    print("\n[10] 文档一致性检查")
    path = BASE / "VERIFICATION.md"
    if not path.exists():
        fail("VERIFICATION.md", "缺失")
        return

    text = path.read_text(encoding="utf-8")

    # 文档里写的自检类数必须与实际注册的检查数一致。
    # 这类"文档说 11 类、实际跑 13 类"的漂移已出现过多次，
    # 会让人误以为某些检查没被执行。
    # 数量直接取自 ALL_CHECKS，不再写死，避免本断言自己变成漂移源。
    actual_checks = len(ALL_CHECKS)
    claimed = re.findall(r"共检查 \*\*(\d+) 类\*\*|检查 (\d+) 类", text)
    claimed_nums = {int(a or b) for a, b in claimed}
    if not claimed_nums:
        fail("自检类数说明", "文档未写明自检检查类数")
    elif claimed_nums != {actual_checks}:
        fail("自检类数说明",
             f"文档写的是 {sorted(claimed_nums)} 类，实际注册 {actual_checks} 类")
    else:
        ok(f"文档写明的自检类数（{actual_checks}）与实际一致")

    # 步骤编号必须唯一且连续
    numbers = [int(m) for m in re.findall(r"^## 第 (\d+) 步", text, re.MULTILINE)]
    if not numbers:
        fail("步骤编号", "未找到任何「第 N 步」标题")
    elif numbers != list(range(len(numbers))):
        dupes = sorted({n for n in numbers if numbers.count(n) > 1})
        fail("步骤编号", f"编号不连续或有重复：{numbers}（重复：{dupes}）")
    else:
        ok(f"步骤编号唯一且连续（0–{len(numbers) - 1}）")

    # 汇总表行数必须与步骤数一致。
    # 注意：必须只统计「## 汇总」那一节里的表格。
    # 文档里还有「实测验收记录」等其它以 "| 数字 " 开头的表格，
    # 全篇统计会把它们一起算进来（曾因此误报 41 行 ≠ 12 步）。
    summary_at = text.find("## 汇总")
    scope = text[summary_at:] if summary_at != -1 else ""
    # 到下一个二级标题为止，避免把后续表格算进汇总表
    nxt = scope.find("\n## ", 1)
    if nxt != -1:
        scope = scope[:nxt]
    table_rows = len(re.findall(r"^\| \d+ ", scope, re.MULTILINE))
    if table_rows == len(numbers):
        ok(f"汇总表覆盖全部 {table_rows} 个步骤")
    else:
        fail("汇总表", f"表中有 {table_rows} 行，但实际有 {len(numbers)} 个步骤")

    # 子小节编号必须归属正确的父步骤（形如 "### 7.1 xxx" 应在 "## 第 7 步" 之后）
    lines = text.splitlines()
    current_step: int | None = None
    bad_subsections: list[str] = []
    for line in lines:
        m_step = re.match(r"^## 第 (\d+) 步", line)
        if m_step:
            current_step = int(m_step.group(1))
            continue
        m_sub = re.match(r"^### (\d+)\.(\d+)", line)
        if m_sub:
            parent = int(m_sub.group(1))
            if current_step is not None and parent != current_step:
                bad_subsections.append(
                    f"「{line.strip()}」出现在第 {current_step} 步之下"
                )
    if bad_subsections:
        fail("子小节编号", "；".join(bad_subsections))
    else:
        ok("子小节编号与所在步骤一致")

    # 每个测试文件都必须在文档的测试表格里出现
    tests_dir = BASE / "tests"
    if tests_dir.exists():
        missing = sorted(
            p.name for p in tests_dir.glob("test_*.py")
            if p.name not in text
        )
        if missing:
            fail("测试表格完整", f"VERIFICATION.md 未提及这些测试文件：{missing}")
        else:
            ok("VERIFICATION.md 已列出全部测试文件")


def check_honesty() -> None:
    """禁止把"未实测"写成"已实测"（计划书 §18 红线）。

    背景：开发主机最初因文件沙箱无法启动子进程，项目一度未真正运行过。
    后来在放开沙箱与迁移后的机器上均已实测（92 项 pytest + 37 项 HTTP
    验收通过，见 LOCAL_RUN.md / VERIFICATION.md）。因此本检查不是禁止
    记录真实结果，而是要求"已实测"式表述必须带可核对的具体证据或否定词。
    """
    print("\n[14] 诚实性检查（不得虚构验证结果）")

    # 这些短语只有在**没有被否定词修饰**时才算违规
    risky = ("实测通过", "已验证", "经实际验证", "测试全部通过", "已跑通")
    negations = (
        "尚未", "未经", "没有", "非实测", "未实测", "不得", "不是",
        # "暂无已验证的评论配图" 是对素材现状的如实描述，属于被否定修饰，
        # 不是"宣称已实测"。缺了这几个词会产生误报。
        "暂无", "尚无", "无已验证", "待", "不需要", "未",
    )

    # 带有具体证据的行不算"虚构"：这类行给出了可核对的实际数值
    # （尺寸、字节数、耗时、HTTP 状态码等）。
    # 本检查要防的是"没跑过却写已通过"，不是禁止记录真实结果。
    evidence = re.compile(
        r"\d+\s*(?:passed|项|字节|MB|KB|B\b)"
        r"|\d{3}\b"                      # HTTP 状态码
        r"|Content-Type|Range|HttpOnly|SameSite"
        r"|shape\s*=|l2_norm|dtype\s*=|PASS"
        r"|\d+\.\d+\s*(?:秒|s\b)"
    )

    offenders: list[str] = []
    # 不扫描 selfcheck.py 自身：它必须**引用**这些短语才能定义规则
    # （risky 元组与解释性文档字符串），扫描它必然产生误报。
    targets = [
        BASE / "README.md",
        BASE / "VERIFICATION.md",
        BASE / "requirements.txt",
    ]
    targets += sorted(
        p for p in BASE.glob("*.py") if p.name != Path(__file__).name
    )

    for path in targets:
        if not path.exists():
            continue
        for line_no, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            for phrase in risky:
                if phrase not in line:
                    continue
                # 同一行里有否定词就放行（例如"尚未实测通过"）
                if any(neg in line for neg in negations):
                    continue
                # 带了具体证据数值的行也放行：这类表述是可核对的实测记录，
                # 不是虚构。真正要拦的是"什么都没有却写已通过"。
                if evidence.search(line):
                    continue
                offenders.append(f"{path.name}:{line_no} 出现「{phrase}」")

    if offenders:
        fail("不得虚构验证结果",
             "以下位置把未实测写成了已实测（计划书 §18 禁止）："
             + "；".join(offenders))
    else:
        ok("文档中的「已实测」式表述均带具体证据或否定修饰，无虚构")


def check_seed_assets() -> None:
    """清单里引用的本地图片必须真实存在。

    这是最容易被忽略、也最容易让人卡住的一类问题：
    CSV 语法完全正确，但图片不在项目内的 data/source_images/ 下，
    预处理会直接报"本地图片必须位于 data/source_images/ 内"。
    """
    print("\n[11] 素材清单与实际文件对照")

    manifest = BASE / "data" / "comment_images_manifest.csv"
    seed = BASE / "data" / "songs_seed.csv"

    if not seed.exists():
        fail("songs_seed.csv", "缺失")
    else:
        text = seed.read_text(encoding="utf-8")
        if "demo_001" in text:
            ok("songs_seed.csv 含兜底歌曲 demo_001")
        else:
            fail("兜底歌曲", "config.FALLBACK_SONG_KEY = demo_001，但清单里没有这一行")

    if not manifest.exists():
        fail("comment_images_manifest.csv", "缺失")
        return

    # ---- 先校验两个 CSV 的字段数是否与表头一致 ----
    # 少写一个逗号不会让 CSV "语法错误"，但会让某一列整体错位，
    # 表现为"音频路径明明写了却读不到"，极难肉眼发现。
    for label, path in (("songs_seed.csv", seed),
                        ("comment_images_manifest.csv", manifest)):
        if not path.exists():
            continue
        try:
            reader = csv.reader(io.StringIO(path.read_text(encoding="utf-8")))
            table = [row for row in reader if row and any(c.strip() for c in row)]
        except Exception as exc:
            fail(f"{label} 无法解析", str(exc))
            continue

        if not table:
            fail(f"{label} 空文件", "没有有效数据行")
            continue

        width = len(table[0])
        bad = [
            (i + 1, len(row))
            for i, row in enumerate(table)
            if len(row) != width
        ]
        if bad:
            detail = "、".join(f"第 {n} 行有 {w} 列" for n, w in bad)
            fail(f"{label} 列数一致", f"表头有 {width} 列，但 {detail}（多为漏写逗号）")
        else:
            ok(f"{label} 共 {len(table) - 1} 数据行，列数均为 {width}")

    rows = list(csv.DictReader(io.StringIO(manifest.read_text(encoding="utf-8"))))
    local_rows = [r for r in rows if (r.get("image_path") or "").strip()]

    if not rows:
        fail("清单内容", "清单为空")
        return

    if not local_rows:
        ok(f"清单共 {len(rows)} 行，全部为远程 URL（无需本地文件）")
        return

    missing: list[str] = []
    outside: list[str] = []
    empty: list[str] = []

    for row in local_rows:
        rel = (row.get("image_path") or "").strip()
        target = (BASE / rel).resolve()

        # 必须位于项目内的 data/source_images/ 下
        try:
            target.relative_to((BASE / "data" / "source_images").resolve())
        except ValueError:
            outside.append(rel)
            continue

        if not target.exists():
            missing.append(rel)
        elif target.stat().st_size == 0:
            empty.append(rel)

    if outside:
        fail("图片路径越界", f"以下路径不在 data/source_images/ 内：{outside}")
    else:
        ok(f"{len(local_rows)} 条本地图片路径全部位于 data/source_images/ 内")

    if missing:
        fail("图片文件缺失",
             f"清单引用了不存在的文件（预处理会失败）：{missing}")
        print("\n    提示：仓库里的真实图片可能在上一级目录，需先复制：")
        print("      New-Item -ItemType Directory -Force -Path "
              '"data\\source_images\\ziyoudeni" | Out-Null')
        print('      Copy-Item "..\\data\\source_images\\ziyoudeni\\*.jpg" '
              '"data\\source_images\\ziyoudeni\\"')
    else:
        ok(f"{len(local_rows)} 个图片文件均已就位")

    if empty:
        fail("图片为空文件", f"以下文件大小为 0，属于占位文件：{empty}")
    elif not missing:
        ok("图片均非空（未使用占位文件）")

    # ---- 音频：清单里声明了就必须真实存在，且不得为空文件 ----
    song_rows = list(
        csv.DictReader(io.StringIO(seed.read_text(encoding="utf-8")))
    ) if seed.exists() else []

    declared = [
        (r.get("song_key", ""), (r.get("audio_local_path") or "").strip())
        for r in song_rows
        if (r.get("audio_local_path") or "").strip()
    ]

    if not declared:
        ok("没有歌曲声明音频路径（页面将显示「音频暂未提供」）")
    else:
        audio_missing: list[str] = []
        audio_empty: list[str] = []
        for key, rel in declared:
            target = (BASE / "static" / rel).resolve()
            if not target.exists():
                audio_missing.append(f"{key}→{rel}")
            elif target.stat().st_size == 0:
                audio_empty.append(f"{key}→{rel}")

        if audio_empty:
            fail("音频为空文件", f"禁止用空文件冒充音频（计划书 §18）：{audio_empty}")
        elif audio_missing:
            fail("音频文件缺失",
                 f"以下歌曲声明了音频但文件不存在：{audio_missing}")
            print("\n    提示：可复制仓库里的真实音频样本：")
            print('      Copy-Item "..\\netease_probe\\downloads\\'
                  'fruit_陈粒_40s_sample.mp3" "static\\audio\\demo_002.mp3"')
        else:
            ok(f"{len(declared)} 个声明的音频文件均已就位且非空")

    # ---- README 的素材现状表必须与真实清单一致 ----
    # 这张表是给评审看的关键说明；若它与 CSV 脱节，
    # 就会出现"文档说不能匹配、实际却能匹配"这类自相矛盾。
    readme = BASE / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        song_keys = [r.get("song_key", "").strip() for r in song_rows]
        missing_keys = [k for k in song_keys if k and f"`{k}`" not in text]
        if missing_keys:
            fail("README 素材表",
                 f"以下 song_key 未出现在 README 的素材现状表中：{missing_keys}")
        else:
            ok(f"README 素材现状表覆盖全部 {len(song_keys)} 个 song_key")


def check_launchers() -> None:
    """启动脚本必须能在默认控制台环境下正常工作。

    坑：.cmd/.ps1 里如果含中文，而文件又不是 UTF-8 with BOM，
    Windows PowerShell 5.1 与 cmd 会按 ANSI 代码页（中文系统为 GBK）
    解读，导致乱码甚至解析失败。
    因此必须提供一个**纯 ASCII** 的兜底启动器。
    """
    print("\n[12] 启动脚本检查")

    # 所有 *_ascii.cmd 都必须是纯 ASCII —— 这是它们存在的唯一理由
    ascii_launchers = sorted(BASE.glob("*_ascii.cmd"))
    if not ascii_launchers:
        fail("ASCII 启动器", "缺少 *_ascii.cmd 系列（编码兜底）")
    for path in ascii_launchers:
        raw = path.read_bytes()
        non_ascii = [b for b in raw if b > 0x7F]
        if non_ascii:
            fail(f"{path.name} ASCII 纯度",
                 f"含 {len(non_ascii)} 个非 ASCII 字节，失去编码兜底作用")
        else:
            ok(f"{path.name} 为纯 ASCII（任何代码页下都能运行）")

    # 中文版 .cmd 应设置 UTF-8 代码页
    for name in ("run_selfcheck.cmd", "stage_assets_ascii.cmd"):
        path = BASE / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "chcp 65001" in text:
            ok(f"{name} 设置了 chcp 65001")
        else:
            fail(f"{name} 代码页", "含中文时应设置 chcp 65001，否则输出乱码")

    zh_cmd = BASE / "run_selfcheck.cmd"
    if zh_cmd.exists():
        text = zh_cmd.read_text(encoding="utf-8", errors="replace")
        # 必须有 :accept 子过程，否则 for 块内变量作用域会出错
        if ":accept" in text:
            ok("run_selfcheck.cmd 使用子过程规避 for 变量作用域问题")
        else:
            fail("for 变量作用域", "缺少 :accept 子过程，PATH 探测可能失效")

    # setup_env.ps1 不能只看 .venv\Scripts\python.exe 是否存在：
    # 从别的电脑拷来的虚拟环境 python.exe 还在，但基础解释器路径已失效，
    # 只判存在性会跳过重建，随后在装 torch 时报出无关的错误信息。
    setup = BASE / "setup_env.ps1"
    if setup.exists():
        text = setup.read_text(encoding="utf-8", errors="replace")
        probes_venv = re.search(r"&\s*\$venvPy\s+-c\s", text)
        handles_cfg = "pyvenv.cfg" in text
        if probes_venv and handles_cfg:
            ok("setup_env.ps1 实际运行 venv 解释器验证可用性（不只看文件存在）")
        else:
            missing = []
            if not probes_venv:
                missing.append("缺少对 $venvPy 的试运行探测")
            if not handles_cfg:
                missing.append("未处理 pyvenv.cfg 指向失效的情况")
            fail("setup_env.ps1 venv 健康检查",
                 "；".join(missing) + "，拷机后可能报出与真实原因无关的安装失败")

    # 三个启动器同样会优先选用 .venv。若它们只判存在性，
    # 用户跑自检时会看到"自检失败"，而真正的原因只是一个用不了的虚拟环境。
    launcher_probe = {
        "run_selfcheck.cmd": r'\.venv\\Scripts\\python\.exe"\s+-c\s',
        "run_selfcheck_ascii.cmd": r'\.venv\\Scripts\\python\.exe"\s+-c\s',
        "run_selfcheck.ps1": r"&\s*\$venv\s+-c\s",
    }
    bad_launchers = []
    for name, pattern in launcher_probe.items():
        path = BASE / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if not re.search(pattern, text):
            bad_launchers.append(name)
    if bad_launchers:
        fail("启动器 venv 健康检查",
             f"以下启动器只判 .venv 是否存在、未试运行验证：{bad_launchers}"
             "（旧 venv 会让自检报出假失败）")
    else:
        ok("三个启动器均在选用 .venv 前实际试运行一次")

    # 脚本中不得硬编码含中文的文件名：
    # 若被以错误代码页解读，中文会变乱码导致路径失配、脚本误报"文件不存在"。
    # 注意只检查可执行行——脚本里可能有注释专门解释"为什么不要硬编码"。
    for name in ("run_selfcheck.ps1", "stage_assets.ps1", "setup_env.ps1",
                 "run_selfcheck.cmd", "stage_assets_ascii.cmd"):
        path = BASE / name
        if not path.exists():
            continue
        offenders = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if stripped.startswith(("#", "REM ", "::")):
                continue
            if "fruit_" in line or "陈粒" in line:
                offenders.append(stripped[:60])
        if offenders:
            fail(f"{name} 中文文件名",
                 f"可执行行中硬编码了含中文的文件名，代码页不匹配时会失配：{offenders}")
        else:
            ok(f"{name} 未硬编码中文文件名")

    # 各启动器都必须排除 Store 占位符
    for name in ("run_selfcheck.cmd", "run_selfcheck_ascii.cmd", "run_selfcheck.ps1"):
        path = BASE / name
        if not path.exists():
            fail(name, "缺失")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "WindowsApps" in text:
            ok(f"{name} 排除 Microsoft Store 占位符")
        else:
            fail(f"{name} 占位符排除", "未排除 WindowsApps，可能选中 0 字节目录存根")

    # 会输出中文的 Python 入口必须做 stdout 编码兜底：
    # Windows 控制台默认为 GBK(936)，print() 中文会抛 UnicodeEncodeError，
    # 导致自检/契约检查还没出结论就崩掉。
    for name in ("selfcheck.py", "clip_encoder.py"):
        path = BASE / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if "_force_utf8_stdout" in text and "reconfigure" in text:
            ok(f"{name} 有 stdout 编码兜底（GBK 控制台下不会崩）")
        else:
            fail(f"{name} 编码兜底",
                 "输出中文但未重配置 stdout，GBK 控制台下会 UnicodeEncodeError")

    # 启动器应为子进程设置 UTF-8
    for name in ("run_selfcheck.cmd", "run_selfcheck_ascii.cmd", "run_selfcheck.ps1"):
        path = BASE / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "PYTHONIOENCODING" in text or "PYTHONUTF8" in text:
            ok(f"{name} 为子进程设置 UTF-8 环境变量")
        else:
            fail(f"{name} UTF-8 环境", "未设置 PYTHONIOENCODING/PYTHONUTF8")

    # ---- 可移植性：项目必须能整体拷到任意目录/另一台机器 ----
    # 代码里出现写死的盘符会让"换台机器运行"直接失败。
    # 注意排除 tests/：那里的 "C:/Windows/..." 是**故意**用来验证
    # 危险路径被拒绝的输入数据，不是真实的机器路径。
    abs_pattern = re.compile(r"['\"][A-Za-z]:[\\/]")
    bad_abs: list[str] = []
    for path in sorted(BASE.rglob("*.py")):
        # 用相对 BASE 的路径判断，避免因项目所在的**上级目录**名
        # 恰好含 "tests"/".venv" 而把全部文件跳过，导致检查静默失效。
        rel_parts = path.relative_to(BASE).parts
        if rel_parts and rel_parts[0] in (".venv", "tests"):
            continue
        for line_no, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            if line.lstrip().startswith("#"):
                continue
            if abs_pattern.search(line):
                bad_abs.append(f"{path.relative_to(BASE)}:{line_no}")
    if bad_abs:
        fail("可移植性（绝对路径）",
             f"以下代码写死了盘符，换机器会失败：{bad_abs}")
    else:
        ok("业务代码中无写死的绝对路径（可整体拷贝；测试夹具中的危险路径样本已排除）")

    # 脚本必须锚定自身位置，不能依赖调用者的当前目录
    for name in ("run_selfcheck.cmd", "run_selfcheck_ascii.cmd",
                 "stage_assets_ascii.cmd"):
        path = BASE / name
        if path.exists() and "%~dp0" in path.read_text(encoding="utf-8", errors="replace"):
            ok(f"{name} 以 %~dp0 锚定自身位置")
        elif path.exists():
            fail(f"{name} 位置锚定", "未使用 %~dp0，依赖调用者当前目录")
    for name in ("run_selfcheck.ps1", "stage_assets.ps1", "setup_env.ps1"):
        path = BASE / name
        if path.exists() and "$PSScriptRoot" in path.read_text(encoding="utf-8",
                                                              errors="replace"):
            ok(f"{name} 以 $PSScriptRoot 锚定自身位置")
        elif path.exists():
            fail(f"{name} 位置锚定", "未使用 $PSScriptRoot，依赖调用者当前目录")


# 检查清单集中定义：文档一致性检查会读取它的长度，
# 避免"文档写 13 类、实际 14 类"这种漂移（此前多次出现）。
ALL_CHECKS: tuple = (
    check_syntax,
    check_prohibited_logic,
    check_offline_guards,
    check_path_safety,
    check_db_constraints,
    check_fallback_boundary,
    check_templates,
    check_deliverables,
    check_test_isolation,
    check_docs_consistency,
    check_seed_assets,
    check_launchers,
    check_undefined_names,
    check_honesty,
)


def main() -> int:
    print("=" * 62)
    print("image2music_demo 静态自检（不需要 torch / flask / 网络）")
    print("=" * 62)

    for check in ALL_CHECKS:
        try:
            check()
        except Exception:
            fail(check.__name__, "检查过程抛出异常\n" + traceback.format_exc())

    print("\n" + "=" * 62)
    print(f"通过 {len(PASSES)} 项，失败 {len(FAILURES)} 项")
    print("=" * 62)

    if FAILURES:
        print("\n失败详情：")
        for item in FAILURES:
            print(f"  - {item}")
        print("\n注意：本脚本只做静态检查，不能替代 pytest 与真实模型自检。")
        return 1

    print("\n静态检查全部通过。")
    print("下一步仍需在装好依赖后执行：")
    print("  1) python clip_encoder.py    # 用真实模型验证 512 维契约")
    print("  2) pytest -v                 # 运行完整测试")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# 🔧 技术实现与第三方资源说明

> 本文档说明：系统用了哪些开源组件与第三方模型、各自的许可证、素材版权边界，
> 以及**明确没有使用**的技术。评委可据此核查合规性与技术选型。
>
> 最后更新：v1.0 冻结版本

---

## 一、技术实现总览

### 1.1 运行时技术栈

| 层 | 技术 | 版本 | 用途 |
|---|---|---|---|
| Web 框架 | Flask | 3.0.3 | 路由、请求处理、Jinja2 模板渲染 |
| 表单安全 | Flask-WTF + WTForms | 1.2.1 / 3.1.2 | 全部 POST 表单 CSRF 防护 |
| 数据库 | SQLite（Python 内置 `sqlite3`） | 3.x | 歌曲、素材向量、留言、点赞、回复 |
| 图片处理 | Pillow | 10.4.0 | 格式校验、EXIF 纠正、缩放、透明通道合成 |
| 数值计算 | NumPy | 2.1.1 | 向量归一化与余弦相似度矩阵乘法 |
| HTTP | Requests | 2.32.3 | 仅预处理阶段下载素材图（运行期不联网） |
| 推理框架 | PyTorch (CPU) | 2.4.1 | 三个模型的推理后端 |
| 模型加载 | Transformers + huggingface-hub + tokenizers + safetensors | 4.44.2 / 0.24.6 / 0.19.1 / 0.4.5 | 本地缓存加载，`local_files_only=True` |
| 测试 | pytest | 8.3.2 | 144 项自动化测试 |

完整的固定版本清单见 [`requirements.txt`](../requirements.txt)。

### 1.2 前端实现

- **原生 HTML / CSS / JavaScript**，无 React/Vue 等框架
- **零 CDN 依赖**：字体、样式、脚本全部本地，断网可完整演示
- 特殊效果（3D 翻面、星光粒子、加载动画）用 CSS transform / animation + Canvas 手写实现
- 图片上传前用 `Canvas` 压缩（`static/js/image-compress.js`），减少传输体积

### 1.3 检索算法（自研部分）

这部分是本项目的核心逻辑，**非第三方组件**：

```text
图片向量 (CLIP 512d)  ──权重 1.0──┐
文本向量 (BGE 512d)   ──权重 0.5──┼─→ RRF 名次融合 (k=60) → Top-1
情绪分值 (StructBERT)  ──权重 0.2──┘        ↑
                              仅在「与最佳图片相似度差 ≤ 0.08」的候选内重排
```

- **RRF（Reciprocal Rank Fusion）**：按名次而非分数融合，规避不同模型相似度尺度不可比的问题
- **相对窗口 0.08**：不是阈值、不是置信度，只是限制文本/情绪的重排范围，保护图片主导性
- 融合结果**不展示、不落库**，页面不出现任何评分或百分比

---

## 二、第三方模型

| 模型 | 提供方 | 用途 | 许可证 | 分发方式 |
|---|---|---|---|---|
| `openai/clip-vit-base-patch32` | OpenAI | 图片特征提取（512 维，L2 归一化） | MIT | 不随仓库分发，首次由使用者自行下载到 `data/model_cache/` |
| `BAAI/bge-small-zh-v1.5` | 北京智源人工智能研究院 (BAAI) | 中文笔记文本语义向量（512 维） | MIT | 同上，见 `scripts/fetch_models.py` |
| `iic/nlp_structbert_sentiment-classification_chinese-base` | 阿里达摩院 (ModelScope) | 中文情感极性打分 | Apache-2.0 | 同上 |

**使用方式说明：**

- 三个模型均**只做推理**，不做微调、不再分发权重
- 运行期通过 `local_files_only=True` + 离线环境变量加载，**不请求任何在线服务**
- 模型许可证以上游模型主页（HuggingFace / ModelScope Model Card）的最新声明为准；
  若你计划商用，请自行复核上游条款
- 模型体积与下载方法见 [`materials.md`](materials.md)

> 本项目**未使用**任何大语言模型（LLM）生成内容或做二次判断，
> 检索结果完全由向量相似度与 RRF 名次决定，可复现、可解释。

---

## 三、第三方素材

| 类型 | 来源 | 版权 | 是否随仓库分发 |
|---|---|---|---|
| 演示音频（`.mp3`） | 音乐平台公开曲目 | 归原作者 / 唱片公司 / 平台所有 | ❌ 否 |
| 素材图片（笔记配图） | 用户在音乐社区公开发布的笔记图片 | 归原发布者所有 | ❌ 否 |
| 演示所用笔记文本 | 用户公开笔记正文 | 归原发布者所有 | ⚠️ 仅以清单形式保存在 CSV（`data/comment_images_manifest.csv` 等） |
| 字体 | 系统字体（华文行楷 / 楷体 / Georgia 等） | 随操作系统授权 | ❌ 否，调用系统已装字体 |
| 图标 / 装饰 | Unicode 字符与自绘 CSS/SVG | 本项目 | ✅ 是 |

**版权边界声明：**

- 所有素材**仅用于本次黑客松的非商业技术演示**，不用于任何商业用途
- 本项目**不包含**任何平台爬虫作为运行依赖；素材由人工整理 CSV 清单后导入
- 如需用于商业场景，请替换为**自有版权或已获授权**的音频与图片：
  修改 `data/songs_seed.csv` 与 `data/comment_images_manifest.csv`，
  重新执行 `python preprocess.py` 即可，**代码中没有任何硬编码的歌曲或图片**
- 详见 [`materials.md`](materials.md) 与项目根目录 [`LICENSE`](../LICENSE)

---

## 四、明确未使用的技术

为避免夸大，这里列出**没有**采用的东西：

| 未使用 | 说明 |
|---|---|
| FAISS / 向量数据库 | 素材仅 352 条，`numpy` 矩阵乘法即可，无需引入额外依赖 |
| 在线 API / 云推理 | 全流程离线，隐私数据不出本机 |
| LLM 二次判断 / 解释生成 | 检索结果由向量决定，不经过语言模型 |
| 多模态融合训练 | 三个模型各自独立推理，仅做名次融合，未联合训练 |
| 音频分析（节奏/频谱） | 音频只用于播放，不参与匹配 |
| 歌词检索 | 未接入歌词数据 |
| 前端框架 / 打包工具 | 原生实现，无需 node_modules 与构建步骤 |
| 用户账号体系 | 仅可选口令门禁，无注册登录与权限模型 |
| 图片质量评分 / 情绪打标展示 | 情绪只用于内部排序，不向用户展示标签 |

---

## 五、合规自查清单

- [x] 仓库不含任何 API Key、口令、Token（已全量扫描）
- [x] 密钥与口令只从环境变量 / `.env` 读取，代码无默认值
- [x] 不含用户隐私数据（数据库、上传图片均被 `.gitignore` 排除）
- [x] 不随仓库分发有版权的音频与图片
- [x] 不使用平台评论接口作为运行依赖
- [x] 不绕过登录、验证码、访问限制或付费墙
- [x] 依赖版本固定，许可证均为常见的宽松开源协议（MIT / BSD / Apache-2.0）
- [x] 公开协议为 MIT，商业化前需自行替换素材

---

## 六、许可证速查

| 依赖 | 许可证 |
|---|---|
| Flask / Werkzeug / Jinja2 | BSD-3-Clause |
| Flask-WTF / WTForms | BSD-3-Clause |
| Pillow | MIT-CMU |
| NumPy | BSD-3-Clause |
| Requests | Apache-2.0 |
| PyTorch | BSD-3-Clause |
| Transformers / huggingface-hub / tokenizers / safetensors | Apache-2.0 |
| pytest | MIT |
| CLIP (模型权重) | MIT |
| BGE-small-zh-v1.5 | MIT |
| StructBERT 中文情感 | Apache-2.0 |

所有许可证均允许在保留版权声明的前提下自由使用、修改与再分发。

"""Flask 应用（计划书 §12、§13、§14）。

启动：python app.py
默认：http://127.0.0.1:5000

单进程、关闭 debug、关闭 reloader、不配置多 worker；每个进程只加载一次 CLIP。
"""
from __future__ import annotations

import hashlib
import io
import logging
import secrets
from datetime import datetime
from functools import wraps
from logging.handlers import RotatingFileHandler

from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_wtf import CSRFProtect
from PIL import Image

import config
import db
import hybrid
import image_utils
from hybrid import PoolUnavailable

logger = logging.getLogger(__name__)

# 会话中保存检索结果的键
SESSION_RESULT_KEY = "result"

# 访客访问密码通过后写入会话的键
ACCESS_SESSION_KEY = "access_ok"


# ================================================================ 应用工厂
def create_app() -> Flask:
    config.ensure_directories()

    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.update(
        SECRET_KEY=config.get_secret_key(),
        MAX_CONTENT_LENGTH=config.MAX_CONTENT_LENGTH,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        # 本地 HTTP 演示不强制 Secure
        SESSION_COOKIE_SECURE=False,
        WTF_CSRF_TIME_LIMIT=None,
        # 本机演示会反复改模板：按 mtime 重新读取，改完刷新页面就生效，
        # 不用重启进程（这不等于 debug/reloader，仍是单进程、关调试）。
        TEMPLATES_AUTO_RELOAD=True,
    )

    CSRFProtect(app)
    _configure_logging(app)

    # ---- 启动时检查：数据库、兜底歌曲、模型与素材索引（各一次） ----
    db_ok = db.check_database()
    if not db_ok:
        logger.error("数据库不可用，Web 将展示友好错误页")
    else:
        # 幂等建表：确保新增的共鸣/回复表在旧库上也会被创建
        try:
            db.init_db()
        except Exception as exc:  # noqa: BLE001
            logger.error("初始化数据库表失败：%s", exc)
            db_ok = False

    encoder = None
    load_failure: str | None = None
    text_encoder = None
    sentiment = None
    text_reason: str | None = None

    if db_ok:
        from clip_encoder import load_encoder_or_none

        encoder, load_failure = load_encoder_or_none()
        if not config.MODEL_DOWNGRADE:
            from text_encoder import load_text_models_or_none

            text_encoder, sentiment, text_reason = load_text_models_or_none()
        else:
            text_reason = "MODEL_DOWNGRADE=True"
    else:
        text_reason = "数据库不可用"

    if db_ok:
        comment_index = hybrid.PoolIndex.build(hybrid.POOL_COMMENT, require_text=True)
    else:
        comment_index = hybrid.PoolIndex.empty()

    service = hybrid.HybridService(
        encoder=encoder,
        text_encoder=text_encoder,
        sentiment=sentiment,
        comment_index=comment_index,
        fallback_song_id=_lookup_fallback_song_id(),
        load_failure_reason=load_failure,
    )

    # 兜底歌曲存在性检查
    fallback_ok = _check_fallback_song()
    if not fallback_ok:
        logger.error("兜底歌曲不存在：%s", config.FALLBACK_SONG_KEY)

    app.extensions["hybrid_service"] = service
    app.extensions["db_ok"] = db_ok
    app.extensions["fallback_ok"] = fallback_ok

    _register_https_redirect(app)
    _register_access_gate(app)
    _register_routes(app)
    _register_error_handlers(app)
    _disable_html_cache(app)

    logger.info(
        "启动完成：数据库=%s 模型=%s 文本=%s 情感=%s 图片笔记组合=%d 兜底歌曲=%s",
        "可用" if db_ok else "不可用",
        "已加载" if encoder is not None else f"降级({load_failure})",
        "可用" if text_encoder is not None else f"不可用({text_reason})",
        "可用" if sentiment is not None else f"不可用({text_reason})",
        service.comment_index_size,
        "存在" if fallback_ok else "缺失",
    )
    return app


def _disable_html_cache(app: Flask) -> None:
    """HTML 一律不缓存：删除留言后刷新必须看到最新列表，而不是浏览器里的旧页。"""

    @app.after_request
    def _no_store(response):
        content_type = response.headers.get("Content-Type", "")
        if content_type.startswith("text/html"):
            response.headers["Cache-Control"] = "no-store, max-age=0"
            response.headers["Pragma"] = "no-cache"
        return response


def _configure_logging(app: Flask) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        handler = RotatingFileHandler(
            config.LOG_DIR / "app.log", maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
        )
        logging.getLogger().addHandler(handler)
    except OSError as exc:
        logger.warning("无法写入日志文件：%s", exc)


def _check_fallback_song() -> bool:
    try:
        with db.get_connection() as conn:
            return db.get_song_by_key(conn, config.FALLBACK_SONG_KEY) is not None
    except Exception as exc:
        logger.error("检查兜底歌曲失败：%s", exc)
        return False


def _lookup_fallback_song_id() -> int | None:
    try:
        with db.get_connection() as conn:
            row = db.get_song_by_key(conn, config.FALLBACK_SONG_KEY)
            return int(row["id"]) if row is not None else None
    except Exception as exc:
        logger.error("读取兜底歌曲 ID 失败：%s", exc)
        return None


def _service() -> hybrid.HybridService:
    from flask import current_app

    return current_app.extensions["hybrid_service"]


def _visitor_key() -> str:
    """本机演示没有账号：用一个会话级随机键实现「同一访客只共鸣一次」。"""
    key = session.get("visitor_key")
    if not key:
        key = secrets.token_urlsafe(16)
        session["visitor_key"] = key
        session.modified = True
    return key


def _format_note_time(ms: object) -> str:
    try:
        value = int(ms)
    except (TypeError, ValueError):
        return ""
    if value <= 0:
        return ""
    try:
        return datetime.fromtimestamp(value / 1000).strftime("%Y-%m-%d")
    except (OverflowError, OSError, ValueError):
        return ""


def _store_matchable_note(
    conn,
    *,
    song_id: int,
    image_local_path: str,
    image_bytes: bytes,
    image_vec,
    content: str,
    nickname: str | None,
    source_link: str | None = None,
) -> int:
    """把用户留下的明信片写入「图片 + 笔记」候选池，供之后的新照片检索。"""
    service = _service()
    digest = hashlib.sha256(image_bytes).hexdigest()

    if image_vec is None:
        embedding = [0.0] * config.EMBEDDING_DIM
        embedding_model = "user-upload"
    else:
        embedding = image_vec
        embedding_model = config.CLIP_MODEL_NAME

    image_id, _ = db.upsert_comment_image(
        conn,
        song_id=song_id,
        image_local_path=image_local_path,
        source_url=None,
        image_sha256=digest,
        embedding=embedding,
        embedding_model=embedding_model,
    )

    text_vec = service.encode_text(content)
    valence = None
    if content and len(content) >= config.EMOTION_MIN_TEXT_LENGTH:
        valence = service.score_text(content)
    db.upsert_item_text(
        conn,
        item_type="comment_image",
        item_id=image_id,
        text=content,
        embedding=text_vec,
        text_model=config.TEXT_MODEL_NAME if text_vec is not None else None,
        valence=valence,
        emotion_label=None,
        emotion_probability=None,
        sentiment_model=config.SENTIMENT_MODEL_NAME if valence is not None else None,
    )
    db.upsert_comment_note(
        conn,
        comment_image_id=image_id,
        note_text=content,
        note_author=(nickname or "").strip() or config.DEFAULT_NOTE_AUTHOR,
        note_time_ms=int(datetime.now().timestamp() * 1000),
        source_link=source_link,
    )
    return image_id


def _image_aspect(relative: str | None) -> float | None:
    """读取静态图片的宽高比，用于让明信片长成图片的形状。"""
    if not relative:
        return None
    absolute = image_utils.static_relative_to_absolute(relative)
    if absolute is None or not absolute.exists():
        return None
    try:
        with Image.open(absolute) as img:
            width, height = img.size
    except Exception:  # noqa: BLE001
        return None
    if width <= 0 or height <= 0:
        return None
    return width / height


# ================================================================ 装饰器
def require_database(view):
    """数据库不可用时返回友好 503 页面。"""

    @wraps(view)
    def wrapper(*args, **kwargs):
        from flask import current_app

        if not current_app.extensions.get("db_ok", False):
            return render_template("error.html",
                                   title="数据库不可用",
                                   message="数据库暂时无法访问，请稍后重试。"), 503
        return view(*args, **kwargs)

    return wrapper


def require_fallback(view):
    """兜底歌曲不存在时返回配置错误页 503。"""

    @wraps(view)
    def wrapper(*args, **kwargs):
        from flask import current_app

        if not current_app.extensions.get("fallback_ok", True):
            return render_template("error.html",
                                   title="配置错误",
                                   message="兜底歌曲未配置或不存在，请联系管理员。"), 503
        return view(*args, **kwargs)

    return wrapper


# ================================================================ HTTPS
def _register_https_redirect(app: Flask) -> None:
    @app.before_request
    def force_public_https():
        if not config.PUBLIC_HOST:
            return None
        host = (request.host or "").split(":", 1)[0].lower()
        if host != config.PUBLIC_HOST.lower():
            return None
        proto = (request.headers.get("X-Forwarded-Proto")
                 or request.headers.get("X-Forwarded-Scheme")
                 or request.scheme)
        if proto.lower() != "https":
            target = "https://" + request.host + request.full_path
            return redirect(target.rstrip("?"), code=301)
        return None


# ================================================================ 访问门禁
def _safe_next(target: str | None) -> str | None:
    if target and target.startswith("/") and not target.startswith("//"):
        return target
    return None


def _register_access_gate(app: Flask) -> None:
    @app.before_request
    def require_access():
        if not config.ACCESS_PASSWORD:
            return None
        endpoint = request.endpoint
        if endpoint in {"access_login", "access_logout"}:
            return None
        # 登录页本身需要 CSS/JS；图片、音频、上传图在登录前一律挡住
        if endpoint == "static" and request.path.startswith(("/static/css/", "/static/js/")):
            return None
        if session.get(ACCESS_SESSION_KEY):
            return None
        nxt = request.full_path.rstrip("?") or "/"
        return redirect(url_for("access_login", next=nxt))

    @app.route("/login", methods=["GET", "POST"])
    def access_login():
        if session.get(ACCESS_SESSION_KEY):
            return redirect(_safe_next(request.args.get("next")) or url_for("index"))

        error = None
        nxt = request.values.get("next") or ""
        if request.method == "POST":
            password = request.form.get("password") or ""
            if config.ACCESS_PASSWORD and secrets.compare_digest(
                password, config.ACCESS_PASSWORD
            ):
                session[ACCESS_SESSION_KEY] = True
                session.modified = True
                return redirect(_safe_next(nxt) or url_for("index"))
            error = "访问密码不正确。"

        return render_template("access_login.html", error=error, next=nxt)

    @app.route("/logout")
    def access_logout():
        session.pop(ACCESS_SESSION_KEY, None)
        session.modified = True
        return redirect(url_for("access_login"))


# ================================================================ 路由
def _register_routes(app: Flask) -> None:
    @app.route("/")
    @require_database
    def index():
        return render_template(
            "index.html",
            max_upload_mib=config.MAX_UPLOAD_FILE_BYTES / (1024 * 1024),
            allowed_formats="JPEG、PNG",
            query_text_max_length=config.QUERY_TEXT_MAX_LENGTH,
            nickname_placeholder=config.NICKNAME_PLACEHOLDER,
            compose_placeholder=config.COMPOSE_PLACEHOLDER,
            nickname_max_length=config.NICKNAME_MAX_LENGTH,
        )

    @app.route("/upload", methods=["POST"])
    @require_database
    @require_fallback
    def upload():
        message = (request.form.get("message") or "").strip()
        if len(message) > config.QUERY_TEXT_MAX_LENGTH:
            flash(f"想写下的话最多 {config.QUERY_TEXT_MAX_LENGTH} 个字符。", "error")
            return redirect(url_for("index"))

        nickname = (request.form.get("nickname") or "").strip()
        if len(nickname) > config.NICKNAME_MAX_LENGTH:
            flash(f"署名最多 {config.NICKNAME_MAX_LENGTH} 个字符。", "error")
            return redirect(url_for("index"))

        # 未选择文件
        file_storage = request.files.get("image")
        if file_storage is None or not file_storage.filename:
            flash("还没有照片——先放一张此刻的照片。", "error")
            return redirect(url_for("index"))

        try:
            raw = file_storage.read()
        except Exception as exc:
            logger.error("读取上传文件失败：%s", exc)
            flash("读取照片失败，请再试一次。", "error")
            return redirect(url_for("index"))

        # 图片校验失败 → 直接拒绝，绝不当作模型故障走兜底
        try:
            normalized = image_utils.normalize_once(raw)
        except image_utils.ImageValidationError as exc:
            flash(str(exc), "error")
            return redirect(url_for("index"))

        # 保存上传图片
        try:
            upload_relative = image_utils.save_upload_image(normalized)
        except image_utils.ImageValidationError as exc:
            flash(str(exc), "error")
            return redirect(url_for("index"))
        except OSError as exc:
            logger.error("保存上传图片失败：%s", exc)
            flash("保存照片失败，请再试一次。", "error")
            return redirect(url_for("index"))

        service = _service()
        token = secrets.token_urlsafe(16)

        # ---- 查询图片向量（模型不可用时只影响歌曲兜底，不影响图片校验） ----
        image_vec = None
        degrade_reason: str | None = None
        try:
            with Image.open(io.BytesIO(normalized)) as img:
                image_vec = service.encode_image(img.convert("RGB"))
        except PoolUnavailable as exc:
            degrade_reason = str(exc)
        except Exception as exc:  # noqa: BLE001
            logger.exception("图片推理失败")
            degrade_reason = f"图片推理失败：{exc}"

        # ---- 文本 / 情感信号（只在留言足够长且图片模型可用时参与） ----
        text_vec = None
        valence = None
        if image_vec is not None and message and len(message) >= config.EMOTION_MIN_TEXT_LENGTH:
            text_vec = service.encode_text(message)
            valence = service.score_text(message)

        # ---- 检索：图片 + 可选留言 → 一条「图片 + 笔记」组合 ----
        outcome = None
        if image_vec is not None:
            try:
                outcome = service.match_song(image_vec, text_vec, valence)
            except Exception:  # noqa: BLE001
                logger.exception("歌曲检索失败")

        if outcome is None:
            fallback_id = service.fallback_song_id()
            if fallback_id is None:
                abort(503)
            stored = {
                "token": token,
                "song_id": int(fallback_id),
                "upload_image_path": upload_relative,
                "matched_image_id": None,
                "query_message": message,
                "query_nickname": nickname,
                "used_text": False,
                "used_emotion": False,
                "degraded": True,
                "degrade_reason": degrade_reason or "素材库没有有效的评论图片向量",
                "post_used": False,
            }
        else:
            stored = {
                "token": token,
                "song_id": int(outcome.item.song_id),
                "upload_image_path": upload_relative,
                "matched_image_id": int(outcome.item.item_id),
                "query_message": message,
                "query_nickname": nickname,
                "used_text": bool(outcome.used_text),
                "used_emotion": bool(outcome.used_emotion),
                "degraded": False,
                "degrade_reason": None,
                "post_used": False,
            }

        # 检索阶段不写库：留言只是排序信号，要不要留到留言墙由用户在结果页决定。
        stored["post_used"] = False

        # 绑定到会话（每会话只保留最新一次结果）
        session[SESSION_RESULT_KEY] = stored
        session.modified = True

        if message:
            flash("已收到你的照片，星空替你找到了一张相似的明信片。", "message")

        # 重定向，避免刷新结果页再次上传
        return redirect(url_for("result"))

    @app.route("/result")
    @require_database
    def result():
        stored = session.get(SESSION_RESULT_KEY)
        # token 与 upload_image_path 下面都会被直接读取，一并校验，
        # 避免会话结构不完整时抛 KeyError（与 /submit_post 保持一致）。
        if (not stored or "song_id" not in stored
                or "upload_image_path" not in stored or "token" not in stored):
            flash("还没有寄出照片，先寄一张吧。", "error")
            return redirect(url_for("index"))

        # 校验失败时保留的输入
        draft = session.pop("draft_post", None)

        # ---------------------------------------------------- 命中「图片 + 笔记」
        song = _get_song_or_404(int(stored["song_id"]))

        matched_image_path = None
        matched_note = None
        matched_note_time = ""
        if stored.get("matched_image_id") is not None:
            image_id = int(stored["matched_image_id"])
            matched_image_path = _lookup_comment_image_path(image_id)
            if matched_image_path:
                try:
                    with db.get_connection() as conn:
                        matched_note = db.get_comment_note_text(conn, image_id)
                except Exception as exc:
                    logger.error("读取评论配文失败：%s", exc)
                if matched_note is not None:
                    matched_note_time = _format_note_time(matched_note["note_time_ms"])

        audio_relative, audio_available = _resolve_audio(song)

        # 明信片的长宽比跟随图片本身，避免出现大片留白或裁切（值域内做夹紧）
        card_aspect = (
            _image_aspect(matched_image_path)
            or _image_aspect(stored["upload_image_path"])
            or 1.5
        )
        card_aspect = round(max(0.6, min(2.0, card_aspect)), 4)

        return render_template(
            "result.html",
            song=song,
            upload_image_path=stored["upload_image_path"],
            matched_image_path=matched_image_path,
            card_aspect=card_aspect,
            matched_note=matched_note,
            matched_note_time=matched_note_time,
            degraded=bool(stored.get("degraded")),
            degrade_reason=stored.get("degrade_reason"),
            token=stored["token"],
            post_used=bool(stored.get("post_used")),
            audio_relative=audio_relative,
            audio_available=audio_available,
            query_message=stored.get("query_message") or "",
            query_nickname=stored.get("query_nickname") or "",
            used_text=bool(stored.get("used_text")),
            used_emotion=bool(stored.get("used_emotion")),
            default_note_author=config.DEFAULT_NOTE_AUTHOR,
            nickname_placeholder=config.NICKNAME_PLACEHOLDER,
            compose_placeholder=config.COMPOSE_PLACEHOLDER,
            draft=draft,
        )

    @app.route("/song/<int:song_id>")
    @require_database
    def song_detail(song_id: int):
        song = _get_song_or_404(song_id)
        visitor_key = _visitor_key()

        note_rows: list = []
        likes: dict[int, int] = {}
        liked: set[int] = set()
        replies: list = []
        post_rows: list = []
        post_likes: dict[int, int] = {}
        post_liked: set[int] = set()
        post_replies: list = []
        try:
            with db.get_connection() as conn:
                note_rows = db.list_song_note_images(conn, song_id)
                likes = db.note_like_counts(conn, song_id)
                image_ids = [int(row["comment_image_id"]) for row in note_rows]
                liked = db.liked_note_ids(conn, visitor_key, image_ids)
                replies = db.note_replies_for_song(conn, song_id)

                post_rows = db.list_posts_for_song(conn, song_id)
                post_ids = [int(row["id"]) for row in post_rows]
                post_likes = db.post_like_counts(conn, song_id)
                post_liked = db.liked_post_ids(conn, visitor_key, post_ids)
                post_replies = db.post_replies_for_song(conn, song_id)
        except Exception as exc:  # noqa: BLE001
            logger.error("读取歌曲空间失败：%s", exc)

        replies_by_image: dict[int, list] = {}
        for reply in replies:
            replies_by_image.setdefault(int(reply["comment_image_id"]), []).append(reply)
        replies_by_post: dict[int, list] = {}
        for reply in post_replies:
            replies_by_post.setdefault(int(reply["post_id"]), []).append(reply)

        note_items = []
        for row in note_rows:
            image_id = int(row["comment_image_id"])
            note_items.append(
                {
                    "image_id": image_id,
                    "image_local_path": row["image_local_path"],
                    "note_text": row["note_text"],
                    "author": (row["note_author"] or "").strip()
                    or config.DEFAULT_NOTE_AUTHOR,
                    "time": _format_note_time(row["note_time_ms"]),
                    "likes": likes.get(image_id, 0),
                    "liked": image_id in liked,
                    "replies": replies_by_image.get(image_id, []),
                    "sort_ms": int(row["note_time_ms"] or 0),
                }
            )

        post_items = []
        for row in post_rows:
            post_id = int(row["id"])
            created_at = str(row["created_at"] or "")
            sort_ms = 0
            try:
                sort_ms = int(
                    datetime.fromisoformat(created_at.replace(" ", "T")).timestamp()
                    * 1000
                )
            except ValueError:
                pass
            post_items.append(
                {
                    "post_id": post_id,
                    "image_local_path": row["upload_image_path"],
                    "content": row["content"],
                    "author": (row["nickname"] or "").strip()
                    or config.DEFAULT_NICKNAME,
                    "time": created_at[:10] if len(created_at) >= 10 else created_at,
                    "likes": post_likes.get(post_id, 0),
                    "liked": post_id in post_liked,
                    "replies": replies_by_post.get(post_id, []),
                    "sort_ms": sort_ms,
                }
            )

        # 统一成一条时间线：用户投递的 post 与素材 note 都是「这首歌下的明信片」。
        card_items = []
        for item in note_items:
            anchor = f"note-{item['image_id']}"
            card_items.append(
                {
                    "kind": "note",
                    "item_id": item["image_id"],
                    "anchor": anchor,
                    "image_local_path": item["image_local_path"],
                    "content": item["note_text"],
                    "author": item["author"],
                    "time": item["time"],
                    "likes": item["likes"],
                    "liked": item["liked"],
                    "replies": item["replies"],
                    "sort_ms": item["sort_ms"],
                    "postcard_url": url_for(
                        "note_postcard", image_id=item["image_id"]
                    ),
                    "like_url": url_for("like_note", image_id=item["image_id"]),
                    "reply_url": url_for("reply_note", image_id=item["image_id"]),
                    "reply_next": url_for(
                        "song_detail", song_id=song_id, _anchor=anchor
                    ),
                }
            )
        for item in post_items:
            anchor = f"post-{item['post_id']}"
            card_items.append(
                {
                    "kind": "post",
                    "item_id": item["post_id"],
                    "anchor": anchor,
                    "image_local_path": item["image_local_path"],
                    "content": item["content"],
                    "author": item["author"],
                    "time": item["time"],
                    "likes": item["likes"],
                    "liked": item["liked"],
                    "replies": item["replies"],
                    "sort_ms": item["sort_ms"],
                    "postcard_url": url_for("postcard", post_id=item["post_id"]),
                    "like_url": url_for(
                        "like_postcard", post_id=item["post_id"]
                    ),
                    "reply_url": url_for(
                        "reply_postcard", post_id=item["post_id"]
                    ),
                    "reply_next": url_for(
                        "song_detail", song_id=song_id, _anchor=anchor
                    ),
                }
            )
        # 同一张明信片可能同时以「素材 note」和「用户 post」存在：
        # 内容、作者、日期都相同时只保留一张，优先保留用户投递的 post。
        deduped: dict[tuple[str, str, str], dict] = {}
        for item in card_items:
            key = (
                " ".join((item["content"] or "").split()),
                (item["author"] or "").strip(),
                (item["time"] or "")[:10],
            )
            existing = deduped.get(key)
            if existing is None or (
                item["kind"] == "post" and existing["kind"] == "note"
            ):
                deduped[key] = item
        card_items = list(deduped.values())

        card_items.sort(
            key=lambda item: (item["sort_ms"], item["item_id"]), reverse=True
        )

        audio_relative, audio_available = _resolve_audio(song)

        return render_template(
            "song_detail.html",
            song=song,
            card_items=card_items,
            default_note_author=config.DEFAULT_NOTE_AUTHOR,
            compose_placeholder=config.COMPOSE_PLACEHOLDER,
            nickname_placeholder=config.NICKNAME_PLACEHOLDER,
            max_upload_mib=config.MAX_UPLOAD_FILE_BYTES // (1024 * 1024),
            content_max_length=config.POST_CONTENT_MAX_LENGTH,
            audio_relative=audio_relative,
            audio_available=audio_available,
        )

    @app.route("/song/<int:song_id>/postcard", methods=["POST"])
    @require_database
    def song_postcard(song_id: int):
        """把一张照片和一句话直接留在当前歌曲的星群里。"""
        song = _get_song_or_404(song_id)

        content = (request.form.get("content") or "").strip()
        if not content:
            flash("先写一句想留在这颗星群里的话。", "error")
            return redirect(url_for("song_detail", song_id=song_id))
        if len(content) > config.POST_CONTENT_MAX_LENGTH:
            flash(
                f"留言最多 {config.POST_CONTENT_MAX_LENGTH} 个字符。",
                "error",
            )
            return redirect(url_for("song_detail", song_id=song_id))

        nickname = (request.form.get("nickname") or "").strip()
        if len(nickname) > config.NICKNAME_MAX_LENGTH:
            flash(f"署名最多 {config.NICKNAME_MAX_LENGTH} 个字符。", "error")
            return redirect(url_for("song_detail", song_id=song_id))

        file_storage = request.files.get("image")
        if file_storage is None or not file_storage.filename:
            flash("先选一张属于此刻的照片。", "error")
            return redirect(url_for("song_detail", song_id=song_id))

        try:
            raw = file_storage.read()
        except Exception as exc:  # noqa: BLE001
            logger.error("读取星群明信片照片失败：%s", exc)
            flash("读取照片失败，请再试一次。", "error")
            return redirect(url_for("song_detail", song_id=song_id))

        try:
            normalized = image_utils.normalize_once(raw)
        except image_utils.ImageValidationError as exc:
            flash(str(exc), "error")
            return redirect(url_for("song_detail", song_id=song_id))

        try:
            image_relative = image_utils.save_upload_image(normalized)
        except image_utils.ImageValidationError as exc:
            flash(str(exc), "error")
            return redirect(url_for("song_detail", song_id=song_id))
        except OSError as exc:
            logger.error("保存星群明信片照片失败：%s", exc)
            flash("保存照片失败，请再试一次。", "error")
            return redirect(url_for("song_detail", song_id=song_id))

        embedding = None
        try:
            with Image.open(io.BytesIO(normalized)) as img:
                embedding = _service().encode_image(img.convert("RGB"))
        except Exception as exc:  # noqa: BLE001
            # 图片已经能正常保存和展示；向量失败不应让整张明信片消失。
            logger.warning("星群明信片图片向量失败，使用占位向量：%s", exc)

        try:
            with db.get_connection() as conn:
                image_id = _store_matchable_note(
                    conn,
                    song_id=int(song["id"]),
                    image_local_path=image_relative,
                    image_bytes=normalized,
                    image_vec=embedding,
                    content=content,
                    nickname=nickname,
                    source_link=None,
                )
        except db.DatabaseUnavailable:
            return render_template(
                "error.html",
                title="数据库不可用",
                message="数据库暂时无法访问，请稍后重试。",
            ), 503
        except Exception as exc:  # noqa: BLE001
            logger.error("写入星群明信片失败：%s", exc)
            flash("这张明信片暂时没放进星群，请再试一次。", "error")
            return redirect(url_for("song_detail", song_id=song_id))

        try:
            _service().refresh_comment_index()
        except Exception as exc:  # noqa: BLE001
            logger.warning("刷新评论图索引失败：%s", exc)

        flash("明信片已经留在这首歌的星群里。", "success")
        return redirect(
            url_for("song_detail", song_id=song_id) + f"#note-{image_id}"
        )

    @app.route("/note/<int:image_id>/like", methods=["POST"])
    @require_database
    def like_note(image_id: int):
        try:
            with db.get_connection() as conn:
                song_id = db.note_song_id(conn, image_id)
                if song_id is None:
                    abort(404)
                liked = db.toggle_note_like(
                    conn,
                    comment_image_id=image_id,
                    visitor_key=_visitor_key(),
                )
                like_count = db.note_like_count(conn, image_id)
        except db.DatabaseUnavailable:
            return render_template("error.html",
                                   title="数据库不可用",
                                   message="数据库暂时无法访问，请稍后重试。"), 503

        wants_json = (
            request.headers.get("X-Requested-With") == "XMLHttpRequest"
            or "application/json" in request.accept_mimetypes
        )
        if wants_json:
            # 不整页刷新，歌曲才不会因为一次共鸣而停下来。
            return jsonify({"liked": liked, "likes": like_count})

        flash("已共鸣。" if liked else "已取消共鸣。", "success")
        return redirect(url_for("song_detail", song_id=song_id) + f"#note-{image_id}")

    @app.route("/note/<int:image_id>/reply", methods=["POST"])
    @require_database
    def reply_note(image_id: int):
        try:
            with db.get_connection() as conn:
                song_id = db.note_song_id(conn, image_id)
        except db.DatabaseUnavailable:
            return render_template("error.html",
                                   title="数据库不可用",
                                   message="数据库暂时无法访问，请稍后重试。"), 503
        if song_id is None:
            abort(404)

        # 回复统一匿名：不读取昵称，统一记「风外客」
        content = (request.form.get("content") or "").strip()
        if not content:
            flash("回信不能为空。", "error")
            return redirect(url_for("song_detail", song_id=song_id) + f"#note-{image_id}")
        if len(content) > config.POST_CONTENT_MAX_LENGTH:
            flash(f"回信最多 {config.POST_CONTENT_MAX_LENGTH} 个字符。", "error")
            return redirect(url_for("song_detail", song_id=song_id) + f"#note-{image_id}")

        try:
            with db.get_connection() as conn:
                db.add_note_reply(
                    conn,
                    comment_image_id=image_id,
                    nickname=config.DEFAULT_NOTE_AUTHOR,
                    content=content,
                )
        except db.DatabaseUnavailable:
            return render_template("error.html",
                                   title="数据库不可用",
                                   message="数据库暂时无法访问，请稍后重试。"), 503

        flash("回信已经放进星群。", "success")
        next_url = (request.form.get("next") or "").strip()
        if next_url.startswith("/") and not next_url.startswith("//"):
            return redirect(next_url)
        return redirect(url_for("song_detail", song_id=song_id) + f"#note-{image_id}")

    @app.route("/postcard/<int:post_id>/like", methods=["POST"])
    @require_database
    def like_postcard(post_id: int):
        try:
            with db.get_connection() as conn:
                if db.get_post_with_song(conn, post_id) is None:
                    abort(404)
                liked = db.toggle_post_like(
                    conn,
                    post_id=post_id,
                    visitor_key=_visitor_key(),
                )
                like_count = db.post_like_count(conn, post_id)
        except db.DatabaseUnavailable:
            return render_template(
                "error.html",
                title="数据库不可用",
                message="数据库暂时无法访问，请稍后重试。",
            ), 503

        wants_json = (
            request.headers.get("X-Requested-With") == "XMLHttpRequest"
            or "application/json" in request.accept_mimetypes
        )
        if wants_json:
            return jsonify({"liked": liked, "likes": like_count})

        flash("已共鸣。" if liked else "已取消共鸣。", "success")
        return redirect(url_for("postcard", post_id=post_id))

    @app.route("/postcard/<int:post_id>/reply", methods=["POST"])
    @require_database
    def reply_postcard(post_id: int):
        try:
            with db.get_connection() as conn:
                if db.get_post_with_song(conn, post_id) is None:
                    abort(404)
        except db.DatabaseUnavailable:
            return render_template(
                "error.html",
                title="数据库不可用",
                message="数据库暂时无法访问，请稍后重试。",
            ), 503

        content = (request.form.get("content") or "").strip()
        if not content:
            flash("回信不能为空。", "error")
            return redirect(url_for("postcard", post_id=post_id))
        if len(content) > config.POST_CONTENT_MAX_LENGTH:
            flash(f"回信最多 {config.POST_CONTENT_MAX_LENGTH} 个字符。", "error")
            return redirect(url_for("postcard", post_id=post_id))

        try:
            with db.get_connection() as conn:
                db.add_post_reply(
                    conn,
                    post_id=post_id,
                    nickname=config.DEFAULT_NOTE_AUTHOR,
                    content=content,
                )
        except db.DatabaseUnavailable:
            return render_template(
                "error.html",
                title="数据库不可用",
                message="数据库暂时无法访问，请稍后重试。",
            ), 503

        flash("回信已经放进星群。", "success")
        next_url = (request.form.get("next") or "").strip()
        if not next_url.startswith("/") or next_url.startswith("//"):
            next_url = url_for("postcard", post_id=post_id)
        return redirect(next_url)

    @app.route("/submit_post", methods=["POST"])
    @require_database
    @require_fallback
    def submit_post():
        stored = session.get(SESSION_RESULT_KEY)
        # upload_image_path 下面会被直接读取，这里一并校验，
        # 避免会话结构不完整时抛出 KeyError（与 /result 保持一致）。
        if (not stored or "token" not in stored or "song_id" not in stored
                or "upload_image_path" not in stored):
            flash("这次寄信已经过期，请重新寄一张照片。", "error")
            return redirect(url_for("index"))

        # 表单 token 必须与会话一致
        form_token = (request.form.get("token") or "").strip()
        if not form_token or not secrets.compare_digest(form_token, stored["token"]):
            flash("这次寄信已经过期，请重新寄一张照片。", "error")
            return redirect(url_for("index"))

        # 当前结果未成功发帖
        if stored.get("post_used"):
            flash("这张明信片已经留在星空，请重新寄一张照片。", "error")
            return redirect(url_for("posts"))

        nickname = (request.form.get("nickname") or "").strip()
        content = (request.form.get("content") or "").strip()

        errors: list[str] = []
        if len(nickname) > config.NICKNAME_MAX_LENGTH:
            errors.append(f"昵称最多 {config.NICKNAME_MAX_LENGTH} 个字符。")
        if not (config.POST_CONTENT_MIN_LENGTH <= len(content) <= config.POST_CONTENT_MAX_LENGTH):
            errors.append(
                f"想留下的话需要在 {config.POST_CONTENT_MIN_LENGTH}～"
                f"{config.POST_CONTENT_MAX_LENGTH} 个字符之间。"
            )
        if errors:
            for message in errors:
                flash(message, "error")
            # 保留输入
            session["draft_post"] = {"nickname": nickname, "content": content}
            return redirect(url_for("result"))

        # 歌曲 ID 与上传图片路径以会话为准，不信任表单隐藏字段
        song_id = int(stored["song_id"])
        upload_image_path = stored["upload_image_path"]

        image_bytes = None
        image_vec = None
        try:
            with db.get_connection() as conn:
                # 歌曲仍然存在
                if not db.song_exists(conn, song_id):
                    flash("这首歌已经不在星空里了，请重新寄一张照片。", "error")
                    return redirect(url_for("index"))

                # 上传图片文件仍然存在
                absolute = image_utils.static_relative_to_absolute(upload_image_path)
                if absolute is None or not absolute.exists():
                    flash("这张照片已经不在了，请重新寄一张。", "error")
                    return redirect(url_for("index"))

                # 图片路径位于 static/uploads/
                if not image_utils.upload_path_is_safe(upload_image_path):
                    logger.warning("拦截非法上传路径：%s", upload_image_path)
                    flash("照片路径异常，请重新寄一张。", "error")
                    return redirect(url_for("index"))

                try:
                    image_bytes = absolute.read_bytes()
                    with Image.open(io.BytesIO(image_bytes)) as img:
                        image_vec = _service().encode_image(img.convert("RGB"))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("留下明信片时图片向量失败：%s", exc)

                if image_bytes is None:
                    flash("读取照片失败，请重新寄一张。", "error")
                    return redirect(url_for("result"))

                # 用户最终选择留下时，只写一张可参与下次检索的图片笔记；
                # 不再写 posts 副本，避免数据库里出现两张一模一样的明信片。
                _store_matchable_note(
                    conn,
                    song_id=song_id,
                    image_local_path=upload_image_path,
                    image_bytes=image_bytes,
                    image_vec=image_vec,
                    content=content,
                    nickname=nickname or config.DEFAULT_NICKNAME,
                    source_link=None,
                )
        except db.DatabaseUnavailable:
            return render_template("error.html",
                                   title="数据库不可用",
                                   message="数据库暂时无法访问，请稍后重试。"), 503

        try:
            _service().refresh_comment_index()
        except Exception as exc:  # noqa: BLE001
            logger.warning("刷新评论图索引失败：%s", exc)

        # 使当前结果的发帖 token 失效，防止重复提交
        stored["post_used"] = True
        session[SESSION_RESULT_KEY] = stored
        session.pop("draft_post", None)
        session.modified = True

        flash("你的明信片已经留在星空。", "success")
        return redirect(url_for("posts"))

    @app.route("/posts")
    @require_database
    def posts():
        page = request.args.get("page", type=int) or 1
        page = max(1, page)
        query = (request.args.get("q") or "").strip()

        try:
            with db.get_connection() as conn:
                total = db.count_wall_cards(conn, query=query or None)
                cards = db.list_wall_cards(
                    conn,
                    page=page,
                    page_size=config.POSTS_PAGE_SIZE,
                    query=query or None,
                )
        except Exception as exc:
            logger.error("读取留言墙失败：%s", exc)
            return render_template("error.html",
                                   title="云层暂时遮住了星空",
                                   message="明信片暂时无法加载，请稍后重试。"), 503

        total_pages = max(1, (total + config.POSTS_PAGE_SIZE - 1) // config.POSTS_PAGE_SIZE)

        return render_template(
            "posts.html",
            cards=cards,
            page=page,
            total_pages=total_pages,
            total=total,
            query=query,
            dev_mode=bool(session.get("dev_mode")),
        )

    @app.route("/postcard/<int:post_id>")
    @require_database
    def postcard(post_id):
        """星空里的一张明信片：照片、留言与它带来的歌。

        从明信片的星空点歌曲名，进入的是这张明信片本身，而不是歌曲空间。
        """
        try:
            with db.get_connection() as conn:
                post = db.get_post_with_song(conn, post_id)
        except Exception as exc:  # noqa: BLE001
            logger.error("读取明信片失败：%s", exc)
            return render_template(
                "error.html",
                title="云层暂时遮住了星空",
                message="这张明信片暂时无法展开，请稍后重试。",
            ), 503

        if post is None:
            abort(404)

        image_path = post["upload_image_path"]
        absolute = image_utils.static_relative_to_absolute(image_path)
        image_available = bool(
            image_utils.upload_path_is_safe(image_path)
            and absolute is not None
            and absolute.exists()
        )
        card_aspect = _image_aspect(image_path) if image_available else None
        if not card_aspect:
            card_aspect = 1.5
        card_aspect = round(max(0.6, min(2.0, card_aspect)), 4)

        created_at = str(post["created_at"] or "")
        posted_date = created_at[:10] if len(created_at) >= 10 else created_at
        audio_relative, audio_available = _resolve_audio(post)

        likes = 0
        liked = False
        replies = []
        try:
            with db.get_connection() as conn:
                likes = db.post_like_count(conn, post_id)
                liked = post_id in db.liked_post_ids(
                    conn, _visitor_key(), [post_id]
                )
                replies = [
                    reply
                    for reply in db.post_replies_for_song(conn, int(post["song_id"]))
                    if int(reply["post_id"]) == post_id
                ]
        except Exception as exc:  # noqa: BLE001
            logger.warning("读取明信片共鸣失败：%s", exc)

        return render_template(
            "postcard.html",
            post=post,
            card_aspect=f"{card_aspect:.4f}",
            image_available=image_available,
            posted_date=posted_date,
            default_author=config.DEFAULT_NICKNAME,
            audio_relative=audio_relative,
            audio_available=audio_available,
            note_id=None,
            card_post_id=post_id,
            likes=likes,
            liked=liked,
            replies=replies,
            compose_placeholder=config.COMPOSE_PLACEHOLDER,
        )

    @app.route("/postcard/note/<int:image_id>")
    @require_database
    def note_postcard(image_id):
        """素材笔记也能展开成同一套明信片，而不是跳到歌曲空间。"""
        try:
            with db.get_connection() as conn:
                note = db.get_note_card_with_song(conn, image_id)
        except Exception as exc:  # noqa: BLE001
            logger.error("读取笔记明信片失败：%s", exc)
            return render_template(
                "error.html",
                title="云层暂时遮住了星空",
                message="这张明信片暂时无法展开，请稍后重试。",
            ), 503

        if note is None:
            abort(404)

        image_path = note["image_local_path"]
        absolute = image_utils.static_relative_to_absolute(image_path)
        image_available = bool(absolute is not None and absolute.exists())
        card_aspect = _image_aspect(image_path) if image_available else None
        if not card_aspect:
            card_aspect = 1.5
        card_aspect = round(max(0.6, min(2.0, card_aspect)), 4)

        posted_date = _format_note_time(note["note_time_ms"])
        if not posted_date:
            created_at = str(note["created_at"] or "")
            posted_date = created_at[:10] if len(created_at) >= 10 else created_at

        author = (note["note_author"] or "").strip() or config.DEFAULT_NOTE_AUTHOR
        post = {
            "upload_image_path": image_path,
            "nickname": author,
            "content": note["note_text"] or "",
            "song_id": int(note["song_id"]),
            "song_name": note["song_name"],
            "artist": note["artist"],
        }
        audio_relative, audio_available = _resolve_audio(note)

        # 同一张 note 的共鸣与回信和歌曲空间共用数据。
        likes = 0
        liked = False
        replies = []
        try:
            with db.get_connection() as conn:
                likes = db.note_like_count(conn, image_id)
                liked = image_id in db.liked_note_ids(
                    conn, _visitor_key(), [image_id]
                )
                replies = [
                    reply
                    for reply in db.note_replies_for_song(conn, int(note["song_id"]))
                    if int(reply["comment_image_id"]) == image_id
                ]
        except Exception as exc:  # noqa: BLE001
            logger.warning("读取明信片共鸣失败：%s", exc)

        return render_template(
            "postcard.html",
            post=post,
            card_aspect=f"{card_aspect:.4f}",
            image_available=image_available,
            posted_date=posted_date,
            default_author=config.DEFAULT_NOTE_AUTHOR,
            audio_relative=audio_relative,
            audio_available=audio_available,
            note_id=image_id,
            card_post_id=None,
            likes=likes,
            liked=liked,
            replies=replies,
            compose_placeholder=config.COMPOSE_PLACEHOLDER,
        )

    # ------------------------------------------------- 开发者模式：删除留言
    @app.route("/dev/login", methods=["POST"])
    @require_database
    def dev_login():
        password = (request.form.get("password") or "").strip()
        # 未配置 DEV_DELETE_PASSWORD 时关闭删除入口，避免空口令即可删除
        if not config.DEV_DELETE_PASSWORD or not secrets.compare_digest(
            password, config.DEV_DELETE_PASSWORD
        ):
            logger.warning("开发者入口：密码错误")
            flash("密码不正确。", "error")
            return redirect(url_for("posts"))

        session["dev_mode"] = True
        session.modified = True
        flash("已进入开发者模式，可以整理明信片。", "success")
        return redirect(url_for("posts"))

    @app.route("/dev/logout", methods=["POST"])
    def dev_logout():
        session.pop("dev_mode", None)
        session.modified = True
        flash("已退出开发者模式。", "success")
        return redirect(url_for("posts"))

    @app.route("/dev/post/<int:post_id>/delete", methods=["POST"])
    @require_database
    def dev_delete_post(post_id: int):
        if not session.get("dev_mode"):
            flash("需要先进入开发者模式。", "error")
            return redirect(url_for("posts"))

        try:
            with db.get_connection() as conn:
                removed = db.delete_post(conn, post_id)
        except db.DatabaseUnavailable:
            return render_template("error.html",
                                   title="数据库不可用",
                                   message="数据库暂时无法访问，请稍后重试。"), 503

        if removed:
            logger.info("开发者模式删除留言：post_id=%s", post_id)
            flash(f"已删除明信片 #{post_id}。", "success")
        else:
            flash("这张明信片不存在，或已经被收走。", "error")

        page = request.form.get("page", type=int) or 1
        return redirect(url_for("posts", page=max(1, page)))


# ================================================================ 辅助
def _get_song_or_404(song_id: int):
    try:
        with db.get_connection() as conn:
            song = db.get_song_by_id(conn, song_id)
    except Exception as exc:
        logger.error("读取歌曲失败：%s", exc)
        abort(503)

    if song is None:
        abort(404)
    return song


def _lookup_comment_image_path(comment_image_id: int) -> str | None:
    try:
        with db.get_connection() as conn:
            row = conn.execute(
                "SELECT image_local_path FROM song_comment_images WHERE id = ?",
                (comment_image_id,),
            ).fetchone()
        if row is None:
            return None
        relative = str(row["image_local_path"])
        absolute = image_utils.static_relative_to_absolute(relative)
        if absolute is None or not absolute.exists():
            return None
        return relative
    except Exception as exc:
        logger.error("读取评论图片失败：%s", exc)
        return None


def _resolve_audio(song) -> tuple[str | None, bool]:
    """返回 (相对路径, 是否可播放)。音频缺失时页面正常显示并提示。"""
    relative = song["audio_local_path"]
    if not relative:
        return None, False

    if not image_utils.audio_path_is_safe(relative):
        logger.warning("音频路径非法：%s", relative)
        return None, False

    absolute = image_utils.static_relative_to_absolute(relative)
    if absolute is None or not absolute.exists():
        return relative, False

    return relative, True


# ================================================================ 错误处理
def _register_error_handlers(app: Flask) -> None:
    @app.errorhandler(404)
    def handle_404(error):
        return render_template("error.html",
                               title="这封信走错了方向",
                               message="你访问的页面或歌曲不在星空里。"), 404

    @app.errorhandler(413)
    def handle_413(error):
        limit_mib = config.MAX_CONTENT_LENGTH / (1024 * 1024)
        return render_template("error.html",
                               title="上传内容过大",
                               message=f"请求体超过限制（最大 {limit_mib:.0f} MiB）。"), 413

    @app.errorhandler(503)
    def handle_503(error):
        return render_template("error.html",
                               title="服务暂时不可用",
                               message="服务暂时不可用，请稍后重试。"), 503

    @app.errorhandler(500)
    def handle_500(error):
        logger.exception("服务器内部错误")
        return render_template("error.html",
                               title="服务器内部错误",
                               message="服务器出现内部错误，请稍后重试。"), 500

    @app.errorhandler(Exception)
    def handle_unexpected(error):
        # 不让 Flask 把原始堆栈展示给用户
        from werkzeug.exceptions import HTTPException

        if isinstance(error, HTTPException):
            return error

        logger.exception("未处理异常")
        return render_template("error.html",
                               title="服务器内部错误",
                               message="服务器出现内部错误，请稍后重试。"), 500


# ================================================================ 入口
def main() -> None:
    """启动 Web 服务：单进程、无 debug、无 reloader。"""
    application = create_app()
    application.run(
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        debug=False,
        use_reloader=False,
        threaded=True,
    )


if __name__ == "__main__":
    main()

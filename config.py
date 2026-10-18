"""集中配置。

所有路径均以项目目录为基准，不依赖启动时的当前工作目录。
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# ------------------------------------------------------------ 本地环境变量
# 密钥与口令一律从环境变量读取，绝不写进仓库。
# 本地开发可把它们放进项目根目录的 .env（已被 .gitignore 排除），
# 模板见 .env.example。


def _load_dotenv(path: Path) -> None:
    """极简 .env 解析：KEY=VALUE，# 开头为注释，已存在的环境变量优先。"""
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(Path(__file__).resolve().parent / ".env")

# ---------------------------------------------------------------- 基础路径
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"

DB_PATH = DATA_DIR / "music.db"
UPLOAD_DIR = STATIC_DIR / "uploads"
COMMENT_IMAGE_DIR = STATIC_DIR / "comment_images"
AUDIO_DIR = STATIC_DIR / "audio"

SOURCE_IMAGES_DIR = DATA_DIR / "source_images"
TEMPLATES_DIR = BASE_DIR / "templates"

SONGS_SEED_CSV = DATA_DIR / "songs_seed.csv"
COMMENT_MANIFEST_CSV = DATA_DIR / "comment_images_manifest.csv"

SECRET_KEY_FILE = DATA_DIR / ".secret_key"
LOG_DIR = DATA_DIR / "logs"

# ---------------------------------------------------------------- CLIP 模型
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_CACHE_DIR = DATA_DIR / "model_cache"
CLIP_DEVICE = "cpu"
EMBEDDING_DIM = 512

# ------------------------------------------------ 中文文本 / 情感模型（新增）
# 用 ModelScope 下载的本地目录；离线加载，运行期不联网。
BGE_MODEL_DIR = DATA_DIR / "model_cache" / "chinese-models" / "bge-small-zh-v1.5"
SENTIMENT_MODEL_DIR = DATA_DIR / "model_cache" / "chinese-models" / "structbert-sentiment"
TEXT_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
SENTIMENT_MODEL_NAME = "iic/nlp_structbert_sentiment-classification_chinese-base"
TEXT_EMBEDDING_DIM = 512

# 文本/情绪参与检索的最短长度：更短的留言区分度太低，只按图片匹配
EMOTION_MIN_TEXT_LENGTH = 6
QUERY_TEXT_MAX_LENGTH = 500

# 混合检索权重（RRF，按名次融合，与相似度尺度无关）
HYBRID_IMAGE_WEIGHT = 1.0
HYBRID_TEXT_WEIGHT = 0.5
HYBRID_EMOTION_WEIGHT = 0.2
HYBRID_RRF_K = 60
# 文本/情绪只在「与最佳图片相似度相差不超过 0.08」的候选间重排；
# 这是相对窗口，不是最低匹配阈值：视觉差距大的候选不会被文字拉上来。
HYBRID_RERANK_IMAGE_MARGIN = 0.08

# 模型降级模式：可用环境变量覆盖（1/true/yes 视为开启）
_DOWNGRADE_RAW = os.environ.get("MODEL_DOWNGRADE", "").strip().lower()
MODEL_DOWNGRADE = _DOWNGRADE_RAW in {"1", "true", "yes", "on"}

FALLBACK_SONG_KEY = "demo_001"

# ---------------------------------------------------------------- 上传限制
MAX_UPLOAD_FILE_BYTES = 8 * 1024 * 1024          # 单文件 8 MiB
MAX_IMAGE_PIXELS = 20_000_000                    # 像素总量上限
MAX_IMAGE_SIDE = 1600                            # 归一化后最长边
ALLOWED_UPLOAD_FORMATS = {"JPEG", "PNG"}

# Flask 请求体上限需为 multipart 开销留余量
MAX_CONTENT_LENGTH = MAX_UPLOAD_FILE_BYTES + (1 * 1024 * 1024)

# ---------------------------------------------------------------- 留言规则
POST_CONTENT_MIN_LENGTH = 1
POST_CONTENT_MAX_LENGTH = 500
NICKNAME_MAX_LENGTH = 30
DEFAULT_NICKNAME = "风外客"
# 笔记作者 / 回复昵称缺省时的显示名（用户可能不愿留下名字）
DEFAULT_NOTE_AUTHOR = "风外客"
NICKNAME_PLACEHOLDER = "写下你的昵称，不留名者，是风外客。"
COMPOSE_PLACEHOLDER = "落笔存下此刻心绪。"
POSTS_PAGE_SIZE = 20

# 公开域名：只对这个域名强制 HTTPS，避免局域网 IP 被误跳转。
# 留空表示不启用该跳转；可用环境变量 PUBLIC_HOST 覆盖。
PUBLIC_HOST = os.environ.get("PUBLIC_HOST", "").strip()

# 访问口令（可选功能）：设置后整站（含媒体文件）需先输入口令才能浏览；
# 留空则关闭访问门禁，适合纯本地演示。
# ⚠️ 口令只从环境变量 / .env 读取，代码中不留任何默认值，避免误上公网时形同虚设。
ACCESS_PASSWORD = os.environ.get("ACCESS_PASSWORD", "").strip()

# 「开发者」删除入口的口令（可选功能）：留空则关闭删除入口。
# 同样只从环境变量 / .env 读取。
DEV_DELETE_PASSWORD = os.environ.get("DEV_DELETE_PASSWORD", "").strip()

# ---------------------------------------------------------------- 素材下载
DOWNLOAD_TIMEOUT_SECONDS = 15
DOWNLOAD_MAX_RETRIES = 3
DOWNLOAD_MAX_BYTES = MAX_UPLOAD_FILE_BYTES

# ---------------------------------------------------------------- 安全
SONG_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")

# 数据库中的静态路径统一相对 static/，且统一使用 "/" 分隔符
STATIC_PATH_PREFIXES = ("audio/", "comment_images/", "uploads/")

SERVER_HOST = (os.environ.get("SERVER_HOST") or "127.0.0.1").strip() or "127.0.0.1"
try:
    SERVER_PORT = int(os.environ.get("SERVER_PORT") or "5000")
except ValueError:
    SERVER_PORT = 5000


def ensure_directories() -> None:
    """创建运行所需目录（幂等）。"""
    for directory in (
        DATA_DIR,
        STATIC_DIR,
        UPLOAD_DIR,
        COMMENT_IMAGE_DIR,
        AUDIO_DIR,
        SOURCE_IMAGES_DIR,
        CLIP_CACHE_DIR,
        LOG_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def is_valid_song_key(song_key: str) -> bool:
    """song_key 仅允许字母、数字、下划线、连字符。"""
    return bool(song_key) and bool(SONG_KEY_PATTERN.match(song_key))


def get_secret_key() -> str:
    """优先读环境变量；否则生成并持久化到 data/.secret_key 复用。"""
    env_key = os.environ.get("FLASK_SECRET_KEY") or os.environ.get("SECRET_KEY")
    if env_key:
        return env_key

    ensure_directories()
    if SECRET_KEY_FILE.exists():
        stored = SECRET_KEY_FILE.read_text(encoding="utf-8").strip()
        if stored:
            return stored

    # 不使用硬编码公开密钥：本地生成随机密钥
    generated = os.urandom(32).hex()
    SECRET_KEY_FILE.write_text(generated, encoding="utf-8")
    try:
        os.chmod(SECRET_KEY_FILE, 0o600)
    except OSError:
        # Windows 上 chmod 语义有限，失败不影响运行
        pass
    return generated

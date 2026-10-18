"""从 ModelScope 拉取中文文本/情感模型（本机访问 HuggingFace 不通时使用）。

用法（项目目录执行）：

    python scripts/fetch_models.py

下载位置：data/model_cache/chinese-models/
- bge-small-zh-v1.5：中文文本向量（512 维），用于留言语义相似度
- structbert-sentiment：StructBERT 中文情感（正面/负面），用于文字情绪

只做一次性准备；运行期 app.py / build_index.py 只读本地目录，不联网。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "model_cache" / "chinese-models"

MODELS = {
    "BAAI/bge-small-zh-v1.5": (
        "bge-small-zh-v1.5",
        [
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "special_tokens_map.json",
        ],
    ),
    "iic/nlp_structbert_sentiment-classification_chinese-base": (
        "structbert-sentiment",
        [
            "config.json",
            "pytorch_model.bin",
            "vocab.txt",
            "label_mapping.json",
            "configuration.json",
        ],
    ),
}

BASE = "https://www.modelscope.cn/api/v1/models/{model}/repo?Revision=master&FilePath={file}"


def download(url: str, target: Path, retries: int = 4) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        print(f"skip {target.name} ({target.stat().st_size} B)", flush=True)
        return
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(
                url, stream=True, timeout=(15, 120), headers={"User-Agent": "Mozilla/5.0"}
            ) as response:
                response.raise_for_status()
                tmp = target.with_suffix(target.suffix + ".part")
                size = 0
                with open(tmp, "wb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                            size += len(chunk)
                tmp.replace(target)
                print(f"done {target.name} {size} B", flush=True)
                return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"retry {attempt} {target.name}: {type(exc).__name__} {exc}", flush=True)
            time.sleep(2 * attempt)
    raise RuntimeError(f"下载失败 {target.name}: {last_error}")


def main() -> int:
    for model, (folder, files) in MODELS.items():
        out = CACHE / folder
        for name in files:
            print(f"== {model} / {name}", flush=True)
            download(BASE.format(model=model, file=name), out / name)
    print("ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

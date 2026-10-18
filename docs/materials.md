# 📦 素材与权限说明

> 代码与素材**分开分发**：仓库里只有源码、模板、测试和截图；
> 模型缓存、音频、素材图片体积大（合计约 2 GB），且有版权归属，**不进 Git 仓库**。
> 按本文三步在本地放好即可完整运行。

## 一、为什么要分开

| 内容 | 体积 | 为什么不入库 |
|---|---|---|
| `data/model_cache/` | ~1.0 GB | 单个权重文件 > 100 MB，超过 GitHub 单文件上限 |
| `static/audio/` | ~600 MB | 演示曲目版权归原作者 / 平台所有 |
| `data/source_images/` | ~126 MB | 平台用户笔记配图，仅作技术演示 |
| `static/comment_images/` | ~116 MB | 由 `preprocess.py` 从原图生成，可复现 |
| `static/uploads/` | 运行期增长 | 用户上传内容 |
| `data/music.db` | 含真实留言 | 本地运行数据 |

`preprocess.py` 是**幂等**的：原图和清单在，就能重新生成 `comment_images/` 与所有向量，
所以素材缺失不会影响代码的可验证性。

---

## 二、准备步骤

### ① 模型：CLIP（图片特征）

首次需要联网，之后 `local_files_only=True` 完全离线：

```bash
python -c "from pathlib import Path; from transformers import CLIPModel, CLIPProcessor; p=Path('data/model_cache').resolve(); CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32', cache_dir=str(p)); CLIPModel.from_pretrained('openai/clip-vit-base-patch32', cache_dir=str(p)); print('CLIP cache ready')"
```

### ② 模型：中文语义 + 情绪

```bash
python scripts/fetch_models.py
```

下载 BGE-small-zh-v1.5（~95 MB）与 StructBERT 中文情感（~390 MB）到 `data/model_cache/chinese-models/`。
> 脚本走 ModelScope（国内可直连）；若你的网络能直连 HuggingFace，也可手动下载到同目录。

### ③ 素材：音频 + 图片清单

1. 把 `.mp3` 放进 `static/audio/`，文件名与 `data/songs_seed.csv` 的 `audio_local_path` 对应；
2. 把图片按歌曲分目录放进 `data/source_images/<song_key>/01.jpg`；
3. 每张图在 `data/comment_images_manifest.csv` 里登记一行：

```csv
song_key,image_url,image_path
demo_001,,data/source_images/demo_001/01.jpg
demo_002,https://example.com/pic.jpg,
```

4. 生成索引：

```bash
python preprocess.py --seed data/songs_seed.csv --image-manifest data/comment_images_manifest.csv
# 换模型或想重算全部：
python preprocess.py --seed data/songs_seed.csv --image-manifest data/comment_images_manifest.csv --rebuild
```

### ④ 只想先看界面？

不用准备任何素材，降级模式直接启动（上传后返回兜底歌曲）：

```powershell
$env:MODEL_DOWNGRADE="1"; python app.py     # Windows PowerShell
```

---

## 三、版权与合规

- 演示曲目、笔记配图版权归**原作者及各音乐平台**所有，仅用于本次黑客松的非商业技术演示。
- 请勿将素材包用于商用、再分发或二次上架。
- 本项目**不包含**任何平台评论爬虫作为必需依赖；素材由人工整理清单导入。
- 如需替换成自己的内容：修改两个 CSV 并重跑 `preprocess.py` 即可，代码没有任何硬编码的歌曲。

---

## 四、素材包获取

素材包（音频 + 原图 + 模型缓存）不随仓库分发，由团队单独提供：

> 📎 **获取方式：** 请向团队成员索取（龚涵述 / 蒋丰屹），或使用仓库 Release 中附带的说明。
> 拿到后直接解压覆盖到项目根目录，保持 `data/` 与 `static/` 的相对路径不变即可。

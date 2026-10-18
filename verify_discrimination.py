r"""判别力验证：把 34 张素材图逐张当查询，看 top-1 是否落回它自己所属的歌曲。

为什么需要这个：预处理报告"有效向量数 = 34"只说明**写进去了**，
不说明**能区分开**。如果两首歌的素材视觉上太像，或向量写错了歌，
匹配就会串——这个脚本专门抓这种情况。

对每张图 i：
  top1_song(i)       := argmax 相似度所在的歌曲
  self_sim(i)        := 图 i 与自己的相似度
  rival_sim(i)       := 图 i 与**其他歌曲**素材的最高相似度（最强干扰）
  边距 margin        := self_sim - rival_sim

判定：
  命中   top1_song(i) == 图 i 所属歌曲
  边距   margin > 0 且足够大（打印分布，边距过小者单独列出）
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import config                                      # noqa: E402
import db                                          # noqa: E402
from clip_encoder import ClipImageEncoder          # noqa: E402
from matcher import MaterialIndex                  # noqa: E402

print('加载 CLIP 模型…')
encoder = ClipImageEncoder.load(config.CLIP_CACHE_DIR)

index = MaterialIndex.build()
n = len(index)
print(f'素材索引：{n} 条向量\n')
if n == 0:
    raise SystemExit('索引为空，无法验证')

# 歌曲名映射
with db.get_connection() as conn:
    songs = {int(r['id']): r for r in conn.execute('select * from songs')}


def song_label(sid):
    r = songs.get(sid)
    return f'{r["song_key"]}/{r["song_name"]}' if r else f'#{sid}'


# ---- 逐张图编码（真实推理，不读库里的向量）----
print('逐张真实推理（不走库里的向量，避免自证）')
queries = []
for i, rec in enumerate(index.records, 1):
    path = config.STATIC_DIR / rec.image_local_path
    with Image.open(path) as im:
        im = im.convert('RGB')
        queries.append(encoder.encode_image(im))
Q = np.vstack(queries).astype(np.float32)
print(f'  完成 {len(queries)} 张\n')

M = index.matrix                     # 已 L2 归一化
S = Q @ M.T                          # (n_query, n_index)

hits, rows = 0, []
for i, rec in enumerate(index.records):
    sims = S[i]
    # 与产品同一原语：argmax 取最高分（不排序、不取 TopN）
    top1 = index.records[int(np.argmax(sims))]
    self_sim = float(sims[i])

    # 最强干扰 = 屏蔽掉同歌素材后的最高相似度
    other = np.array([index.records[j].song_id != rec.song_id for j in range(n)])
    rival = float(sims[other].max()) if other.any() else float('nan')
    margin = self_sim - rival

    ok = top1.song_id == rec.song_id
    hits += ok
    rows.append({
        'i': i, 'img': rec.image_local_path, 'song': rec.song_id,
        'top1_song': top1.song_id, 'ok': ok,
        'self_sim': self_sim, 'rival_sim': rival, 'margin': margin,
    })

print(f'{"图":<44}{"所属歌曲":<26}{"top1 命中":<22}{"自相似":>8}{"最强干扰":>10}{"边距":>8}')
print('-' * 124)
for r in rows:
    mark = '✅' if r['ok'] else '❌'
    print(f'{r["img"][-42:]:<44}{song_label(r["song"])[:24]:<26}'
          f'{mark} {song_label(r["top1_song"])[:16]:<16}'
          f'{r["self_sim"]:>8.4f}{r["rival_sim"]:>10.4f}{r["margin"]:>8.4f}')

margins = np.array([r['margin'] for r in rows])
print()
print(f'命中率        : {hits}/{n} = {hits/n:.1%}')
print(f'边距 min/中位/均值: {margins.min():.4f} / {np.median(margins):.4f} / {margins.mean():.4f}')

weak = [r for r in rows if r['margin'] < 0.05]
if weak:
    print(f'\n⚠️ 边距 < 0.05 的 {len(weak)} 张（区分度偏低，匹配可能不稳）：')
    for r in weak:
        print(f'   {r["img"][-40:]:<42} 边距 {r["margin"]:.4f}  '
              f'被 {song_label(r["top1_song"])} 干扰')
else:
    print('\n✅ 全部素材边距 ≥ 0.05，区分度充足')

if hits == n:
    print('\n结论： PASS —— 每张素材图均正确落回自己所属歌曲')
else:
    print(f'\n结论： FAIL —— {n - hits} 张图被匹配到了别的歌曲')
    sys.exit(1)

# 长期部署（Linux 服务器）

这套文件用于把 Demo 放到一台 7×24 运行的 Linux 服务器上，
由 gunicorn + Caddy 提供 HTTPS 和访问控制。

> 目标机器建议：Ubuntu 22.04/24.04，**2 核 4 GB 内存**、40 GB 磁盘以上。
> 单 worker 是刻意的：CLIP + bge + 情感模型常驻约 1.2 GB，
> 多开 worker 会成倍占内存，而 CPU 推理并不会因为多 worker 变快多少。

## 1. 服务器准备

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3-pip sqlite3 caddy git rsync
```

## 2. 上传项目

把整个项目目录同步到服务器 `/srv/image2music`，**必须包含**：

- `data/music.db`
- `data/model_cache/`（约 1.1 GB）
- `data/source_images/`
- `static/audio/`（约 600 MB）
- `static/comment_images/`
- `static/uploads/`（如果已有用户上传）
- `data/.secret_key`（保留它，否则所有人的登录会话会失效）

示例：

```bash
rsync -av --progress ./image2music_demo/ user@server:/srv/image2music/
```

## 3. 安装依赖

```bash
cd /srv/image2music
python3.12 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install gunicorn==23.0.0
```

模型缓存已在项目里，启动时 `local_files_only=True`，不需要联网。

## 4. 用 systemd 常驻

代码里没有任何默认口令，先把口令写进只有 root 可读的环境文件：

```bash
sudo tee /etc/image2music.env >/dev/null <<'EOF'
FLASK_SECRET_KEY=请替换为随机串
ACCESS_PASSWORD=请替换为访客口令
DEV_DELETE_PASSWORD=请替换为开发者口令
EOF
sudo chmod 600 /etc/image2music.env
```

```bash
sudo cp deploy/image2music.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now image2music
sudo systemctl status image2music
```

默认监听 `127.0.0.1:8000`，只允许本机反向代理访问。

## 5. Caddy 反向代理 + HTTPS + 登录

1. 把域名 A 记录指向服务器 IP；
2. 改 `deploy/Caddyfile` 里的域名和用户名；
3. 生成密码哈希：

```bash
caddy hash-password --plaintext '你的密码'
```

把输出填进 `Caddyfile` 的 `basic_auth`；
4. 安装配置：

```bash
sudo cp deploy/Caddyfile /etc/caddy/Caddyfile
sudo systemctl reload caddy
```

Caddy 会自动申请 Let's Encrypt 证书，之后访问 `https://你的域名`。

## 6. 备份（建议每天）

```bash
chmod +x deploy/backup.sh
sudo crontab -e
# 加入：
# 30 3 * * * /srv/image2music/deploy/backup.sh >> /var/log/image2music-backup.log 2>&1
```

备份内容：SQLite 数据库 + `static/uploads/`，保留 14 天。

## 7. 安全清单

- 口令只写在 `/etc/image2music.env`（`chmod 600`），不要提交到仓库；
- 公网必须开 Caddy `basic_auth`，否则任何人都能发帖、点赞、回复；
- 只开放 22 / 80 / 443，`8000` 不要暴露到公网；
- 定期看磁盘：`df -h`、`du -sh static/uploads`；
- 升级前先备份。

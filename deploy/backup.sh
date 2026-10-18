#!/usr/bin/env bash
# 每日备份：SQLite（在线安全备份）+ 用户上传图片，保留 14 天。
set -euo pipefail

APP_DIR="${APP_DIR:-/srv/image2music}"
BACKUP_DIR="${BACKUP_DIR:-/srv/backups}"
KEEP_DAYS="${KEEP_DAYS:-14}"
STAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$BACKUP_DIR"

sqlite3 "$APP_DIR/data/music.db" ".backup '$BACKUP_DIR/music_$STAMP.db'"
tar -czf "$BACKUP_DIR/uploads_$STAMP.tar.gz" -C "$APP_DIR/static" uploads

find "$BACKUP_DIR" -type f -mtime "+$KEEP_DAYS" -delete

echo "[$(date '+%F %T')] backup done: $BACKUP_DIR/music_$STAMP.db"

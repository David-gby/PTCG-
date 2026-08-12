#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "请使用 sudo 运行本脚本。" >&2
  exit 2
fi

WORKSPACE=/var/lib/cardscope/platform_workspace
PRIVATE="${WORKSPACE}/private"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_ROOT="/var/backups/cardscope/${STAMP}"
install -d -o root -g root -m 0700 "${BACKUP_ROOT}"

if [[ -f "${PRIVATE}/platform.sqlite3" ]]; then
  sqlite3 "${PRIVATE}/platform.sqlite3" ".backup '${BACKUP_ROOT}/platform.sqlite3'"
  sqlite3 "${BACKUP_ROOT}/platform.sqlite3" 'PRAGMA integrity_check;' | tee "${BACKUP_ROOT}/integrity_check.txt"
fi
for item in access_links.json auto_training; do
  if [[ -e "${PRIVATE}/${item}" ]]; then
    cp -a "${PRIVATE}/${item}" "${BACKUP_ROOT}/"
  fi
done
tar -C "$(dirname "${BACKUP_ROOT}")" -czf "${BACKUP_ROOT}.tar.gz" "$(basename "${BACKUP_ROOT}")"
sha256sum "${BACKUP_ROOT}.tar.gz" > "${BACKUP_ROOT}.tar.gz.sha256"
echo "核心状态备份：${BACKUP_ROOT}.tar.gz"
echo "图片与标注文件请同时使用云盘快照或对象存储做异地备份。"

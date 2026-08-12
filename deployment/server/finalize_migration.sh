#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "请使用 sudo 运行本脚本。" >&2
  exit 2
fi

SOURCE="${1:-}"
TARGET=/var/lib/cardscope/platform_workspace
if [[ -z "${SOURCE}" || ! -d "${SOURCE}" ]]; then
  echo "用法: sudo bash finalize_migration.sh <暂存的platform_workspace目录>" >&2
  exit 2
fi
SOURCE="$(realpath "${SOURCE}")"
if [[ ! -f "${SOURCE}/private/platform.sqlite3" || ! -f "${SOURCE}/private/access_links.json" ]]; then
  echo "迁移目录缺少 platform.sqlite3 或 access_links.json。" >&2
  exit 2
fi

CHECK="$(sqlite3 "${SOURCE}/private/platform.sqlite3" 'PRAGMA integrity_check;')"
if [[ "${CHECK}" != "ok" ]]; then
  echo "数据库完整性检查失败：${CHECK}" >&2
  exit 1
fi

if systemctl is-active --quiet cardscope; then
  bash /opt/cardscope/current/deployment/server/backup_state.sh || true
  systemctl stop cardscope
fi
install -d -o cardscope -g cardscope -m 0750 "${TARGET}"
rsync -a "${SOURCE}/" "${TARGET}/"
chown -R cardscope:cardscope "${TARGET}"
systemctl start cardscope

for _ in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:8765/api/platform/v1/health >/dev/null; then
    echo "迁移完成，服务健康检查通过。"
    exit 0
  fi
  sleep 1
done
journalctl -u cardscope -n 120 --no-pager || true
echo "迁移后服务未通过健康检查。" >&2
exit 1

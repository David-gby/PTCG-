#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "请使用 sudo 运行本脚本。" >&2
  exit 2
fi

SOURCE_DIR="${1:-}"
CURRENT_LINK="${CARDSCOPE_CURRENT_LINK:-/opt/cardscope/current}"
WORKSPACE_ROOT="${CARDSCOPE_WORKSPACE:-/var/lib/cardscope/platform_workspace}"
SERVICE_NAME="${CARDSCOPE_SERVICE:-cardscope}"
HEALTH_URL="${CARDSCOPE_HEALTH_URL:-http://127.0.0.1:8765/api/platform/v1/health}"
VENV_PYTHON="${CARDSCOPE_VENV_PYTHON:-/opt/cardscope/venv/bin/python}"
if [[ -z "${SOURCE_DIR}" || ! -f "${SOURCE_DIR}/VERSION" ]]; then
  echo "用法: sudo bash apply_v0.8.1_update.sh <解压后的 CardScope_Application_v0.8.1_20260813 目录>" >&2
  exit 2
fi

SOURCE_DIR="$(realpath "${SOURCE_DIR}")"
CURRENT_DIR="$(readlink -f "${CURRENT_LINK}" || true)"
if [[ -z "${CURRENT_DIR}" || ! -d "${CURRENT_DIR}" ]]; then
  echo "没有找到 ${CURRENT_LINK}，请先确认服务器现有部署；如目录不同可设置 CARDSCOPE_CURRENT_LINK。" >&2
  exit 2
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_DIR="/opt/cardscope/backups/pre_v0.8.1_${STAMP}"
install -d -o cardscope -g cardscope -m 0750 "${BACKUP_DIR}"
cp -a "${CURRENT_DIR}/VERSION" "${CURRENT_DIR}/web" "${CURRENT_DIR}/platform_app" \
  "${CURRENT_DIR}/ml_backend" "${BACKUP_DIR}/"

rsync -a --delete --exclude '__pycache__' --exclude '*.pyc' \
  "${SOURCE_DIR}/web/" "${CURRENT_DIR}/web/"
rsync -a --delete --exclude '__pycache__' --exclude '*.pyc' \
  "${SOURCE_DIR}/platform_app/" "${CURRENT_DIR}/platform_app/"
rsync -a --delete --exclude '__pycache__' --exclude '*.pyc' \
  "${SOURCE_DIR}/ml_backend/" "${CURRENT_DIR}/ml_backend/"
cp -a "${SOURCE_DIR}/VERSION" "${CURRENT_DIR}/VERSION"
chown -R cardscope:cardscope "${CURRENT_DIR}"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "没有找到服务器 Python：${VENV_PYTHON}；如路径不同可设置 CARDSCOPE_VENV_PYTHON。" >&2
  exit 2
fi
"${VENV_PYTHON}" -m pip install -r "${CURRENT_DIR}/requirements.txt"

# Persistent data is intentionally outside the release directory and is never
# overwritten (default: /var/lib/cardscope/platform_workspace).
sudo -u cardscope test -w "${WORKSPACE_ROOT}/private"
systemctl restart "${SERVICE_NAME}"

for _ in $(seq 1 60); do
  if curl -fsS "${HEALTH_URL}" >/dev/null; then
    echo "CardScope v0.8.1 更新完成。"
    echo "备份目录：${BACKUP_DIR}"
    exit 0
  fi
  sleep 1
done

echo "健康检查失败，正在恢复更新前程序文件。" >&2
rsync -a --delete "${BACKUP_DIR}/web/" "${CURRENT_DIR}/web/"
rsync -a --delete "${BACKUP_DIR}/platform_app/" "${CURRENT_DIR}/platform_app/"
rsync -a --delete "${BACKUP_DIR}/ml_backend/" "${CURRENT_DIR}/ml_backend/"
cp -a "${BACKUP_DIR}/VERSION" "${CURRENT_DIR}/VERSION"
chown -R cardscope:cardscope "${CURRENT_DIR}"
systemctl restart "${SERVICE_NAME}"
journalctl -u "${SERVICE_NAME}" -n 120 --no-pager || true
exit 1

#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "请使用 sudo 运行本脚本。" >&2
  exit 2
fi

BUNDLE="${1:-}"
PUBLIC_URL="${2:-}"
if [[ -z "${BUNDLE}" || -z "${PUBLIC_URL}" || ! -f "${BUNDLE}" ]]; then
  echo "用法: sudo bash update_release.sh <新发布zip> <https://正式域名>" >&2
  exit 2
fi
if [[ "${PUBLIC_URL}" != https://* ]]; then
  echo "公网地址必须以 https:// 开头。" >&2
  exit 2
fi

TMP_DIR="$(mktemp -d /tmp/cardscope-release.XXXXXX)"
trap 'rm -rf "${TMP_DIR}"' EXIT
unzip -q "$(realpath "${BUNDLE}")" -d "${TMP_DIR}"
APP_SOURCE="$(find "${TMP_DIR}" -type f -name platform_server.py -printf '%h\n' | head -n 1)"
if [[ -z "${APP_SOURCE}" || ! -f "${APP_SOURCE}/VERSION" ]]; then
  echo "压缩包中没有找到完整的 CardScope 应用。" >&2
  exit 2
fi

PREVIOUS="$(readlink -f /opt/cardscope/current || true)"
VERSION="$(tr -d '\r\n ' < "${APP_SOURCE}/VERSION")"
RELEASE_ID="${VERSION}_$(date -u +%Y%m%dT%H%M%SZ)"
RELEASE_DIR="/opt/cardscope/releases/${RELEASE_ID}"
install -d -o cardscope -g cardscope -m 0750 "${RELEASE_DIR}"
rsync -a --delete --exclude '__pycache__' --exclude '*.pyc' "${APP_SOURCE}/" "${RELEASE_DIR}/"
chown -R cardscope:cardscope "${RELEASE_DIR}"

/opt/cardscope/venv/bin/python -m pip install -r "${RELEASE_DIR}/requirements.txt"
printf 'CARDSCOPE_PUBLIC_BASE_URL=%s\nOMP_NUM_THREADS=4\nMKL_NUM_THREADS=4\nCUDA_VISIBLE_DEVICES=-1\n' "${PUBLIC_URL%/}" > /etc/cardscope/cardscope.env
chown root:cardscope /etc/cardscope/cardscope.env
chmod 0640 /etc/cardscope/cardscope.env

ln -sfn "${RELEASE_DIR}" /opt/cardscope/current
systemctl restart cardscope

for _ in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:8765/api/platform/v1/health >/dev/null; then
    echo "更新完成：${RELEASE_DIR}"
    echo "旧版本仍保留：${PREVIOUS:-无}"
    exit 0
  fi
  sleep 1
done

echo "新版本健康检查失败，正在回滚。" >&2
if [[ -n "${PREVIOUS}" && -d "${PREVIOUS}" ]]; then
  ln -sfn "${PREVIOUS}" /opt/cardscope/current
  systemctl restart cardscope
fi
journalctl -u cardscope -n 120 --no-pager || true
exit 1

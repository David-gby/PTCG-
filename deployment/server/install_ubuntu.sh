#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "请使用 sudo 运行本脚本。" >&2
  exit 2
fi

APP_SOURCE="${1:-}"
PUBLIC_URL="${2:-}"
if [[ -z "${APP_SOURCE}" || -z "${PUBLIC_URL}" ]]; then
  echo "用法: sudo bash install_ubuntu.sh <01_application目录> <https://正式域名>" >&2
  exit 2
fi
APP_SOURCE="$(realpath "${APP_SOURCE}")"
if [[ ! -f "${APP_SOURCE}/platform_server.py" ]]; then
  echo "应用目录不正确：没有找到 platform_server.py" >&2
  exit 2
fi
if [[ "${PUBLIC_URL}" != https://* ]]; then
  echo "正式环境必须使用 https:// 开头的公网地址。" >&2
  exit 2
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y python3 python3-venv python3-pip curl unzip rsync sqlite3 ca-certificates libgl1 libglib2.0-0

python3 - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("CardScope 需要 Python 3.10 或更高版本")
PY

if ! id cardscope >/dev/null 2>&1; then
  useradd --system --home-dir /var/lib/cardscope --create-home --shell /usr/sbin/nologin cardscope
fi

install -d -o cardscope -g cardscope -m 0750 /opt/cardscope/releases
install -d -o cardscope -g cardscope -m 0750 /var/lib/cardscope/platform_workspace
install -d -o cardscope -g cardscope -m 0750 /var/cache/cardscope/runtime /var/cache/cardscope/yolo /var/cache/cardscope/matplotlib
install -d -o root -g cardscope -m 0750 /etc/cardscope

VERSION="$(tr -d '\r\n ' < "${APP_SOURCE}/VERSION")"
RELEASE_ID="${VERSION}_$(date -u +%Y%m%dT%H%M%SZ)"
RELEASE_DIR="/opt/cardscope/releases/${RELEASE_ID}"
install -d -o cardscope -g cardscope -m 0750 "${RELEASE_DIR}"
rsync -a --delete --exclude '__pycache__' --exclude '*.pyc' "${APP_SOURCE}/" "${RELEASE_DIR}/"
chown -R cardscope:cardscope "${RELEASE_DIR}"

if [[ ! -x /opt/cardscope/venv/bin/python ]]; then
  python3 -m venv /opt/cardscope/venv
fi
/opt/cardscope/venv/bin/python -m pip install --upgrade pip wheel
/opt/cardscope/venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
/opt/cardscope/venv/bin/python -m pip install -r "${RELEASE_DIR}/requirements.txt"

printf 'CARDSCOPE_PUBLIC_BASE_URL=%s\nOMP_NUM_THREADS=4\nMKL_NUM_THREADS=4\nCUDA_VISIBLE_DEVICES=-1\n' "${PUBLIC_URL%/}" > /etc/cardscope/cardscope.env
chown root:cardscope /etc/cardscope/cardscope.env
chmod 0640 /etc/cardscope/cardscope.env

install -m 0644 "${RELEASE_DIR}/deployment/server/cardscope.service.template" /etc/systemd/system/cardscope.service
ln -sfn "${RELEASE_DIR}" /opt/cardscope/current
systemctl daemon-reload
systemctl enable --now cardscope

for _ in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:8765/api/platform/v1/health >/dev/null; then
    echo "CardScope 应用服务安装成功。"
    echo "下一步：配置 Caddy/Nginx 的 HTTPS 反向代理，然后迁移正式数据。"
    exit 0
  fi
  sleep 1
done

systemctl status cardscope --no-pager || true
journalctl -u cardscope -n 120 --no-pager || true
echo "服务未通过健康检查，请根据上面的日志处理。" >&2
exit 1

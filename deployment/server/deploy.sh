#!/usr/bin/env bash
# CardScope 一键部署脚本（在本地开发机运行）
# 把本地代码 + 模型一键发布到服务器生产环境。
#
# 功能：
#   1. git push 代码到服务器（server 远端）
#   2. 用 scp 传输模型文件（git 传不了大模型，走 scp）
#   3. 服务器重启 cardscope-home 服务
#   4. 健康检查验证
#
# 用法：
#   bash deploy.sh                 # 代码 + 模型 全量发布
#   bash deploy.sh --code-only     # 只推代码，不传模型
#   bash deploy.sh --model-only    # 只传模型，不推代码
#   bash deploy.sh --no-restart    # 推送但不重启服务
#
# 依赖：
#   - 本地已配好 server 远端（ubuntu@134.175.83.65:/home/ubuntu/cardscope）
#   - 本地已配好免密 SSH（~/.ssh/id_ed25519）
#   - 模型目录：本脚本同级目录 ../ml_backend/models/

set -Eeuo pipefail

# ===== 可配置项 =====
SERVER="ubuntu@134.175.83.65"
SERVER_PATH="/home/ubuntu/cardscope"
MODELS_SRC="$(cd "$(dirname "$0")/../.." && pwd)/ml_backend/models"
MODELS_DST="${SERVER_PATH}/ml_backend/models"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE="${REMOTE:-server}"
BRANCH="${BRANCH:-main}"
HEALTH_URL="http://127.0.0.1:8765/api/platform/v1/health"

# ===== 参数解析 =====
DO_CODE=1
DO_MODELS=1
DO_RESTART=1
for arg in "$@"; do
  case "$arg" in
    --code-only)  DO_MODELS=0 ;;
    --model-only) DO_CODE=0 ;;
    --no-restart) DO_RESTART=0 ;;
    *) echo "未知参数: $arg（支持 --code-only / --model-only / --no-restart）" >&2; exit 2 ;;
  esac
done

# ===== 工具函数 =====
step() { echo ""; echo "==> $1"; }

remote_cmd() { ssh -i "$SSH_KEY" -o BatchMode=yes -o ConnectTimeout=10 "$SERVER" "$1"; }

# ===== 1. 推送代码 =====
if [[ "$DO_CODE" -eq 1 ]]; then
  step "[1/3] 推送代码到服务器 (${REMOTE} ${BRANCH})"
  # 先检查是否有未提交的改动
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "⚠️  工作区有未提交的改动，请先 commit。" >&2
    git status --short >&2
    exit 1
  fi
  git push "$REMOTE" "$BRANCH"
  echo "✅ 代码已推送：$(git rev-parse --short HEAD)"
fi

# ===== 2. 传输模型 =====
if [[ "$DO_MODELS" -eq 1 ]]; then
  step "[2/3] 传输模型到服务器"
  if [[ ! -d "$MODELS_SRC" ]]; then
    echo "⚠️  未找到模型目录: $MODELS_SRC" >&2
    exit 1
  fi
  scp -i "$SSH_KEY" -o BatchMode=yes "$MODELS_SRC"/*.pt "$SERVER":"$MODELS_DST"
  echo "✅ 模型已传输"
fi

# ===== 3. 重启服务并验证 =====
if [[ "$DO_RESTART" -eq 1 ]]; then
  step "[3/3] 重启服务并健康检查"
  remote_cmd "sudo systemctl restart cardscope-home"
  # 等待服务就绪
  for _ in $(seq 1 30); do
    if remote_cmd "curl -fsS -m 5 $HEALTH_URL" >/dev/null 2>&1; then
      echo "✅ 健康检查通过：$(remote_cmd "curl -fsS $HEALTH_URL")"
      exit 0
    fi
    sleep 2
  done
  echo "❌ 健康检查失败，查看日志：" >&2
  remote_cmd "sudo journalctl -u cardscope-home -n 50 --no-pager" >&2 || true
  exit 1
fi

echo ""
echo "✅ 完成（未重启服务，如需生效请手动执行: ssh $SERVER 'sudo systemctl restart cardscope-home'）"

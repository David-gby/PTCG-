from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from platform_app.http_server import build_server
from platform_app.service import PlatformService
from studio.config import load_app_config
from studio.store import StudioStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CardScope enterprise centering platform")
    parser.add_argument("--host", default="127.0.0.1", help="127.0.0.1 for local use; 0.0.0.0 for LAN/reverse proxy")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--workspace", type=Path, default=ROOT / "platform_workspace")
    parser.add_argument("--public-base-url", default="")
    parser.add_argument("--instance-id", default="")
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    studio_workspace = workspace / "studio_data"
    config = load_app_config(ROOT / "studio_config.json", workspace_override=studio_workspace)
    store = StudioStore(config)
    platform_config = json.loads((ROOT / "platform_config.json").read_text(encoding="utf-8"))
    release_version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    service = PlatformService(
        store,
        workspace,
        pass_deviation_percent=float(platform_config["centering"]["pass_deviation_percent"]),
    )
    server = build_server(
        service,
        args.host,
        args.port,
        ROOT / "web",
        config.max_upload_bytes,
        config.max_json_bytes,
        instance_id=args.instance_id.strip(),
        release_version=release_version,
    )
    actual_host, actual_port = server.server_address[:2]
    browser_host = "127.0.0.1" if actual_host in {"0.0.0.0", "::"} else actual_host
    base_url = args.public_base_url.strip() or f"http://{browser_host}:{actual_port}"
    access = service.initialize_access(base_url)
    service.start_batch_worker()

    print("\nCardScope 平台已启动")
    print(f"企业检测链接：{access['enterprise_url']}")
    print(f"内部管理链接：{access['admin_url']}")
    print(f"访问凭据文件：{service.access_file}")
    if args.host == "0.0.0.0" and not args.public_base_url:
        print("提示：当前仅生成本机地址。对外发布前请配置 HTTPS 域名和反向代理。")
    print("按 Ctrl+C 停止平台。\n")

    if not args.no_browser and os.environ.get("CARDSCOPE_NO_BROWSER") != "1":
        try:
            webbrowser.open(access["enterprise_url"])
        except Exception:
            pass
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nCardScope 平台已停止。")
    finally:
        server.server_close()
        service.stop_batch_worker()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

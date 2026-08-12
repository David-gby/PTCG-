from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from platform_app.auto_training import run_worker
from platform_app.service import PlatformService
from studio.config import load_app_config
from studio.store import StudioStore


def main() -> int:
    parser = argparse.ArgumentParser(description="CardScope safe automatic training worker")
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    job_path = args.job.resolve()
    try:
        import json

        job = json.loads(job_path.read_text(encoding="utf-8-sig"))
        workspace = Path(job["workspace_root"]).resolve()
    except (OSError, ValueError, KeyError) as exc:
        print(f"Invalid automatic training job: {exc}", file=sys.stderr)
        return 2
    config = load_app_config(ROOT / "studio_config.json", workspace_override=workspace / "studio_data")
    service = PlatformService(StudioStore(config), workspace)
    return run_worker(service, job_path)


if __name__ == "__main__":
    raise SystemExit(main())

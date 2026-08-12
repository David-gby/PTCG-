from __future__ import annotations

import copy
import base64
import csv
import hashlib
import hmac
import io
import json
import math
import re
import secrets
import shutil
import sqlite3
import sys
import tempfile
import threading
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from studio.errors import StudioError
from studio.measurements import centering_measurements
from studio.security import atomic_json
from studio.store import StudioStore

from .database import PlatformDatabase, json_text, utc_now
from .result_exports import inspection_result_path, write_inspection_result
from .reference_registration import RegistrationInputError, run_reference_registration


SAFE_ID = re.compile(r"^[a-z][a-z0-9_]{4,80}$")
USERNAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{2,63}$")
PASSWORD_SCRYPT_N = 2**14
PASSWORD_SCRYPT_R = 8
PASSWORD_SCRYPT_P = 1
SESSION_HOURS = 12
LOGIN_LOCK_AFTER = 5
LOGIN_LOCK_SECONDS = 15 * 60
MAX_BATCH_FILES = 500
MAX_BATCH_FILE_BYTES = 100 * 1024 * 1024
MAX_BATCH_TOTAL_BYTES = 5 * 1024 * 1024 * 1024
BATCH_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "application/octet-stream"}
DISPLAY_IMAGE_VERSION = "v1"
DISPLAY_PREVIEW_MAX_DIMENSION = 1600
DISPLAY_WEBP_QUALITY = 90
ISSUE_TAGS = {
    "outer_frame_wrong",
    "inner_frame_wrong",
    "inner_left_wrong",
    "inner_right_wrong",
    "inner_top_wrong",
    "inner_bottom_wrong",
    "glare_or_reflection",
    "shadow_interference",
    "perspective_extreme",
    "card_cropped",
    "other",
}


class PlatformError(Exception):
    def __init__(self, status: int, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = details or {}


def _password_hash(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=PASSWORD_SCRYPT_N,
        r=PASSWORD_SCRYPT_R,
        p=PASSWORD_SCRYPT_P,
        dklen=32,
    )
    return "scrypt${}${}${}${}${}".format(
        PASSWORD_SCRYPT_N,
        PASSWORD_SCRYPT_R,
        PASSWORD_SCRYPT_P,
        base64.urlsafe_b64encode(salt).decode("ascii").rstrip("="),
        base64.urlsafe_b64encode(digest).decode("ascii").rstrip("="),
    )


def _password_matches(password: str, encoded: str) -> bool:
    try:
        scheme, raw_n, raw_r, raw_p, raw_salt, raw_digest = encoded.split("$", 5)
        if scheme != "scrypt":
            return False
        salt = base64.urlsafe_b64decode(raw_salt + "=" * (-len(raw_salt) % 4))
        expected = base64.urlsafe_b64decode(raw_digest + "=" * (-len(raw_digest) % 4))
        actual = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=int(raw_n),
            r=int(raw_r),
            p=int(raw_p),
            dklen=len(expected),
        )
    except (TypeError, ValueError):
        return False
    return hmac.compare_digest(actual, expected)


def _generated_password() -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789!@#$%"
    while True:
        value = "".join(secrets.choice(alphabet) for _ in range(16))
        if all(any(character in group for character in value) for group in (
            "ABCDEFGHJKLMNPQRSTUVWXYZ", "abcdefghijkmnopqrstuvwxyz", "23456789", "!@#$%"
        )):
            return value


def _model_version(prelabel: dict[str, Any]) -> str:
    generator = prelabel.get("generator") if isinstance(prelabel.get("generator"), dict) else {}
    return str(generator.get("algorithm_version") or generator.get("version") or "unknown")


def _confidence(prelabel: dict[str, Any]) -> dict[str, float | None]:
    stages = prelabel.get("stages") if isinstance(prelabel.get("stages"), dict) else {}
    outer = stages.get("outer") if isinstance(stages.get("outer"), dict) else {}
    inner = stages.get("inner") if isinstance(stages.get("inner"), dict) else {}
    outer_value = outer.get("confidence")
    inner_value = inner.get("yolo_confidence")
    try:
        outer_float = round(float(outer_value), 6)
    except (TypeError, ValueError):
        outer_float = None
    try:
        inner_float = round(float(inner_value), 6)
    except (TypeError, ValueError):
        inner_float = None
    available = [value for value in (outer_float, inner_float) if value is not None]
    return {
        "outer": outer_float,
        "inner": inner_float,
        "overall": round(min(available), 6) if available else None,
    }


def _compact_prediction(prelabel: dict[str, Any], pass_deviation: float) -> dict[str, Any]:
    centers = prelabel.get("inner_line_centers_px")
    pair = None
    if isinstance(centers, dict):
        pair = centering_measurements(centers, 630, 880).get("centering_pair_percent")
    maximum_deviation = None
    passed = False
    if isinstance(pair, dict):
        maximum_deviation = round(max(abs(float(pair[key]) - 50.0) for key in ("left", "right", "top", "bottom")), 4)
        passed = maximum_deviation <= pass_deviation
    return {
        "status": prelabel.get("status"),
        "reason_codes": list(prelabel.get("reason_codes") or []),
        "prelabel_sha256": prelabel.get("prelabel_sha256"),
        "model_version": _model_version(prelabel),
        "outer_corners": copy.deepcopy(prelabel.get("outer_corners")),
        "inner_lines_rectified": copy.deepcopy(prelabel.get("inner_lines_rectified")),
        "inner_line_centers_px": copy.deepcopy(centers),
        "centering_pair_percent": copy.deepcopy(pair),
        "maximum_deviation_percent": maximum_deviation,
        "centering_passed": passed,
        "confidence": _confidence(prelabel),
        "rectified_size": {"width": 630, "height": 880},
    }


def _validate_centers(value: Any) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"left", "right", "top", "bottom"}:
        raise PlatformError(422, "INVALID_INNER_CORRECTION", "内框修正必须包含左、右、上、下四条线。")
    result: dict[str, float] = {}
    for key in ("left", "right", "top", "bottom"):
        item = value[key]
        if isinstance(item, bool):
            raise PlatformError(422, "INVALID_INNER_CORRECTION", "内框修正坐标格式不正确。")
        try:
            result[key] = round(float(item), 4)
        except (TypeError, ValueError) as exc:
            raise PlatformError(422, "INVALID_INNER_CORRECTION", "内框修正坐标格式不正确。") from exc
    if not (0 <= result["left"] < result["right"] <= 629):
        raise PlatformError(422, "INVALID_INNER_CORRECTION", "左右内框线的顺序或范围不正确。")
    if not (0 <= result["top"] < result["bottom"] <= 879):
        raise PlatformError(422, "INVALID_INNER_CORRECTION", "上下内框线的顺序或范围不正确。")
    return result


def _validate_outer(value: Any, source_size: Any) -> list[list[float]] | None:
    if value is None:
        return None
    if not isinstance(source_size, dict):
        raise PlatformError(422, "INVALID_OUTER_CORRECTION", "缺少原图尺寸，无法校验外框修正。")
    try:
        width = float(source_size["width"])
        height = float(source_size["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PlatformError(422, "INVALID_OUTER_CORRECTION", "原图尺寸格式不正确。") from exc
    if not isinstance(value, list) or len(value) != 4:
        raise PlatformError(422, "INVALID_OUTER_CORRECTION", "外框修正必须包含左上、右上、右下、左下四个点。")
    points: list[list[float]] = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise PlatformError(422, "INVALID_OUTER_CORRECTION", "外框点坐标格式不正确。")
        try:
            x_value, y_value = float(point[0]), float(point[1])
        except (TypeError, ValueError) as exc:
            raise PlatformError(422, "INVALID_OUTER_CORRECTION", "外框点坐标格式不正确。") from exc
        if not math.isfinite(x_value) or not math.isfinite(y_value):
            raise PlatformError(422, "INVALID_OUTER_CORRECTION", "外框点坐标必须是有限数值。")
        if not (0 <= x_value <= width - 1 and 0 <= y_value <= height - 1):
            raise PlatformError(422, "INVALID_OUTER_CORRECTION", "外框点超出了原图范围。")
        points.append([round(x_value, 3), round(y_value, 3)])
    crosses: list[float] = []
    for index in range(4):
        current = points[index]
        next_point = points[(index + 1) % 4]
        after = points[(index + 2) % 4]
        crosses.append(
            (next_point[0] - current[0]) * (after[1] - next_point[1])
            - (next_point[1] - current[1]) * (after[0] - next_point[0])
        )
    if any(abs(value) < 1e-6 for value in crosses) or not (
        all(value > 0 for value in crosses) or all(value < 0 for value in crosses)
    ):
        raise PlatformError(422, "INVALID_OUTER_CORRECTION", "外框四边形必须保持凸形且不能交叉。")
    area = abs(
        sum(
            points[index][0] * points[(index + 1) % 4][1]
            - points[(index + 1) % 4][0] * points[index][1]
            for index in range(4)
        )
    ) / 2.0
    if area < width * height * 0.01:
        raise PlatformError(422, "INVALID_OUTER_CORRECTION", "外框区域过小，请重新标注。")
    return points


def _lines_from_centers(centers: dict[str, float]) -> dict[str, list[list[float]]]:
    return {
        "left": [[centers["left"], 0.0], [centers["left"], 879.0]],
        "right": [[centers["right"], 0.0], [centers["right"], 879.0]],
        "top": [[0.0, centers["top"]], [629.0, centers["top"]]],
        "bottom": [[0.0, centers["bottom"]], [629.0, centers["bottom"]]],
    }


class PlatformService:
    def __init__(
        self,
        store: StudioStore,
        workspace: Path,
        *,
        pass_deviation_percent: float = 5.0,
    ) -> None:
        self.store = store
        self.workspace = workspace.resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.private_root = self.workspace / "private"
        self.private_root.mkdir(parents=True, exist_ok=True)
        self.database = PlatformDatabase(self.private_root / "platform.sqlite3")
        self.access_file = self.private_root / "access_links.json"
        self.pass_deviation_percent = float(pass_deviation_percent)
        self.public_base_url = ""
        self._auto_training_lock = threading.Lock()
        self.display_cache_root = self.private_root / "display_cache"
        self.display_cache_root.mkdir(parents=True, exist_ok=True)
        self._display_cache_lock = threading.Lock()
        self.batch_spool_root = self.private_root / "upload_batches"
        self.batch_spool_root.mkdir(parents=True, exist_ok=True)
        self.reference_spool_root = self.private_root / "reference_jobs"
        self.reference_spool_root.mkdir(parents=True, exist_ok=True)
        self._batch_upload_lock = threading.Lock()
        self._batch_worker_lock = threading.Lock()
        self._batch_worker_wake = threading.Event()
        self._batch_worker_stop = threading.Event()
        self._batch_worker: threading.Thread | None = None

    def _read_access(self) -> dict[str, Any]:
        if not self.access_file.is_file():
            raise PlatformError(
                500,
                "ACCESS_FILE_MISSING",
                "平台访问凭据文件丢失。请保留数据库并由管理员执行凭据重置。",
            )
        try:
            access = json.loads(self.access_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise PlatformError(500, "ACCESS_FILE_INVALID", "平台访问凭据文件无法读取。") from exc
        if not isinstance(access, dict):
            raise PlatformError(500, "ACCESS_FILE_INVALID", "平台访问凭据文件格式错误。")
        return access

    def _write_access(self, access: dict[str, Any]) -> None:
        access["schema_version"] = "2.0"
        access["updated_at"] = utc_now()
        atomic_json(self.access_file, access)

    def _share_url(self, token: str | None) -> str | None:
        if not token:
            return None
        return f"{self.public_base_url}/enterprise?access={token}"

    def _login_url(self) -> str:
        return f"{self.public_base_url}/login"

    def _admin_login_url(self) -> str:
        return f"{self.public_base_url}/admin-login"

    @staticmethod
    def _normalize_username(value: Any) -> str:
        username = str(value or "").strip().lower()
        if not USERNAME_PATTERN.fullmatch(username):
            raise PlatformError(
                422,
                "INVALID_USERNAME",
                "登录账号需为 3 至 64 位小写字母、数字、点、下划线或短横线。",
            )
        return username

    @staticmethod
    def _validate_password(value: Any, username: str) -> str:
        password = str(value or "")
        if len(password) < 10 or len(password) > 128:
            raise PlatformError(422, "INVALID_PASSWORD", "密码长度需为 10 至 128 个字符。")
        if password.casefold() == username.casefold():
            raise PlatformError(422, "INVALID_PASSWORD", "密码不能与登录账号相同。")
        return password

    def _unique_username(self) -> str:
        for _ in range(20):
            candidate = f"company_{secrets.token_hex(4)}"
            if (
                self.database.tenant_auth_record(candidate) is None
                and self.database.admin_user_auth_record(candidate) is None
            ):
                return candidate
        raise PlatformError(500, "USERNAME_GENERATION_FAILED", "暂时无法生成企业登录账号，请重试。")

    def _unique_admin_username(self) -> str:
        for _ in range(20):
            candidate = f"annotator_{secrets.token_hex(3)}"
            if (
                self.database.tenant_auth_record(candidate) is None
                and self.database.admin_user_auth_record(candidate) is None
            ):
                return candidate
        raise PlatformError(500, "USERNAME_GENERATION_FAILED", "暂时无法生成标注管理员账号，请重试。")

    @staticmethod
    def _parse_utc(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _future_expiry(days: int, current: str | None = None) -> str:
        now = datetime.now(timezone.utc).replace(microsecond=0)
        parsed = PlatformService._parse_utc(current)
        start = parsed if parsed and parsed > now else now
        return (start + timedelta(days=days)).isoformat().replace("+00:00", "Z")

    def initialize_access(self, public_base_url: str) -> dict[str, Any]:
        base = public_base_url.rstrip("/")
        self.public_base_url = base
        if self.database.empty():
            enterprise_token = "ent_" + secrets.token_urlsafe(32)
            admin_token = "adm_" + secrets.token_urlsafe(32)
            project = self.store.create_project(
                {
                    "name": "企业回传检测",
                    "description": "CardScope 企业端居中度检测与问题样本回传",
                    "default_classification": {
                        "face": "front",
                        "layout_id": "gx_current",
                        "orientation_degrees_cw": 0,
                        "card_type": "",
                    },
                    "custom_metadata": {"platform_managed": True},
                }
            )
            tenant = self.database.create_tenant(
                "示例企业",
                project["id"],
                enterprise_token,
                expires_at=self._future_expiry(365),
            )
            self.database.create_admin_token("平台管理员", admin_token)
            access = {
                "schema_version": "2.0",
                "generated_at": utc_now(),
                "enterprise_token": enterprise_token,
                "enterprise_tokens": {tenant["id"]: enterprise_token},
                "admin_token": admin_token,
                "tenant_id": tenant["id"],
            }
            self._write_access(access)
        elif self.access_file.is_file():
            access = self._read_access()
        else:
            raise PlatformError(
                500,
                "ACCESS_FILE_MISSING",
                "平台访问凭据文件丢失。请保留数据库并由管理员执行凭据重置。",
            )
        tokens = access.get("enterprise_tokens")
        if not isinstance(tokens, dict):
            tokens = {}
        legacy_tenant_id = str(access.get("tenant_id") or "")
        legacy_token = str(access.get("enterprise_token") or "")
        if legacy_tenant_id and legacy_token:
            tokens.setdefault(legacy_tenant_id, legacy_token)
        access["enterprise_tokens"] = tokens
        default_enterprise_token = str(access.get("enterprise_token") or "")
        result = {
            **access,
            "enterprise_url": f"{base}/login",
            "legacy_enterprise_url": (
                f"{base}/enterprise?access={default_enterprise_token}"
                if default_enterprise_token
                else None
            ),
            "admin_url": f"{base}/admin?access={access['admin_token']}",
            "admin_login_url": f"{base}/admin-login",
        }
        self._write_access(result)
        return result

    def authenticate(self, token: str) -> dict[str, Any]:
        principal = self.database.authenticate(token)
        if principal is None:
            raise PlatformError(401, "INVALID_ACCESS_LINK", "访问链接无效或已停用。")
        return principal

    def authenticate_session(self, session_token: str) -> dict[str, Any]:
        principal = self.database.authenticate_session(session_token)
        if principal is None:
            raise PlatformError(401, "SESSION_EXPIRED", "登录状态已失效，请重新登录。")
        return principal

    def login_enterprise(self, payload: Any) -> dict[str, Any]:
        """Authenticate either an enterprise account or a password-based admin account."""
        if not isinstance(payload, dict):
            raise PlatformError(422, "INVALID_LOGIN", "请输入账号和密码。")
        username = self._normalize_username(payload.get("username"))
        password = str(payload.get("password") or "")
        if not password or len(password) > 128:
            raise PlatformError(401, "LOGIN_FAILED", "账号或密码错误。")
        tenant_record = self.database.tenant_auth_record(username)
        admin_record = self.database.admin_user_auth_record(username)
        record = tenant_record or admin_record
        account_kind = "enterprise" if tenant_record else "admin"
        now = datetime.now(timezone.utc).replace(microsecond=0)
        locked_until = self._parse_utc(str(record.get("login_locked_until") or "")) if record else None
        if locked_until and locked_until > now:
            remaining = max(1, int((locked_until - now).total_seconds()))
            raise PlatformError(
                429,
                "LOGIN_LOCKED",
                "连续登录失败次数过多，请稍后再试。",
                {"retry_after_seconds": remaining},
            )
        encoded = str(record.get("password_hash") or "") if record else ""
        # Unknown accounts still perform scrypt work to reduce account-enumeration timing signals.
        if record is None:
            _password_matches(password, _password_hash("CardScope-dummy-password"))
            raise PlatformError(401, "LOGIN_FAILED", "账号或密码错误。")
        if not encoded or not _password_matches(password, encoded):
            recorder = (
                self.database.record_login_failure
                if account_kind == "enterprise"
                else self.database.record_admin_login_failure
            )
            new_lock = recorder(str(record["id"]), lock_after=LOGIN_LOCK_AFTER, lock_seconds=LOGIN_LOCK_SECONDS)
            if new_lock:
                raise PlatformError(
                    429,
                    "LOGIN_LOCKED",
                    "连续登录失败次数过多，账号已临时锁定 15 分钟。",
                    {"retry_after_seconds": LOGIN_LOCK_SECONDS},
                )
            raise PlatformError(401, "LOGIN_FAILED", "账号或密码错误。")
        expiry = self._parse_utc(str(record.get("expires_at") or "")) if account_kind == "enterprise" else None
        if not bool(record.get("active")) or (expiry and expiry <= now):
            raise PlatformError(403, "ACCOUNT_UNAVAILABLE", "账号已停用或过期，请联系平台主管理员。")
        token = "ses_" + secrets.token_urlsafe(36)
        session_expiry = (now + timedelta(hours=SESSION_HOURS)).isoformat().replace("+00:00", "Z")
        if account_kind == "enterprise":
            self.database.record_login_success(str(record["id"]))
            self.database.create_enterprise_session(str(record["id"]), token, session_expiry)
        else:
            self.database.record_admin_login_success(str(record["id"]))
            self.database.create_admin_session(str(record["id"]), token, session_expiry)
        principal = self.database.authenticate_session(token)
        assert principal is not None
        return {
            "session_token": token,
            "expires_at": session_expiry,
            "session": self.session(principal),
        }

    def logout_enterprise(self, session_token: str) -> None:
        self.database.revoke_session(session_token)

    def _require_deletion_idle(self) -> None:
        from .auto_training import active_job

        job = active_job(self.private_root)
        if job is not None:
            raise PlatformError(
                409,
                "TRAINING_JOB_ACTIVE",
                "自动训练正在运行，请等待训练结束后再删除数据。",
                {"job_id": job.get("id")},
            )

    @staticmethod
    def require_role(principal: dict[str, Any], role: str) -> None:
        if principal.get("role") != role:
            raise PlatformError(403, "ROLE_FORBIDDEN", "当前访问链接没有执行此操作的权限。")

    @staticmethod
    def admin_access_level(principal: dict[str, Any]) -> str:
        if principal.get("role") != "admin":
            return ""
        return str(principal.get("admin", {}).get("access_level") or "owner")

    @classmethod
    def require_owner(cls, principal: dict[str, Any]) -> None:
        cls.require_role(principal, "admin")
        if cls.admin_access_level(principal) != "owner":
            raise PlatformError(403, "OWNER_PERMISSION_REQUIRED", "此操作仅限平台主管理员。")

    @classmethod
    def require_annotation_access(cls, principal: dict[str, Any]) -> None:
        cls.require_role(principal, "admin")
        if cls.admin_access_level(principal) not in {"owner", "annotator"}:
            raise PlatformError(403, "ANNOTATION_PERMISSION_REQUIRED", "当前账号没有人工标注权限。")

    def session(self, principal: dict[str, Any]) -> dict[str, Any]:
        if principal["role"] == "enterprise":
            tenant = principal["tenant"]
            return {
                "role": "enterprise",
                "display_name": tenant["name"],
                "tenant_id": tenant["id"],
                "expires_at": tenant.get("expires_at"),
                "limits": {"rectified_width": 630, "rectified_height": 880},
                "policy": {
                    "pass_deviation_percent": self.pass_deviation_percent,
                    "feedback_requires_manual_review": True,
                },
            }
        access_level = self.admin_access_level(principal)
        owner = access_level == "owner"
        return {
            "role": "admin",
            "display_name": principal["admin"]["label"],
            "username": principal["admin"].get("username"),
            "access_level": access_level,
            "auth_type": principal["admin"].get("auth_type", "token"),
            "permissions": {
                "annotate_feedback": True,
                "manage_enterprises": owner,
                "manage_admins": owner,
                "view_all_inspections": owner,
                "export_data": owner,
                "manage_training": owner,
                "delete_data": owner,
            },
            "summary": self.database.summary(),
        }

    def _approved_training_feedback_ids(self) -> list[str]:
        return [
            str(item["id"])
            for item in self.database.list_feedback(status="approved", limit=1000)
            if item.get("exported_feedback_id")
        ]

    def admin_training_status(self, principal: dict[str, Any]) -> dict[str, Any]:
        self.require_owner(principal)
        from .auto_training import active_deployment, active_job, active_model_manifest, gpu_status, list_jobs, load_settings

        settings = load_settings(self.private_root)
        jobs = list_jobs(self.private_root)
        approved_ids = self._approved_training_feedback_ids()
        latest_snapshot = max((int(job.get("approved_snapshot_count", 0)) for job in jobs), default=0)
        new_count = max(0, len(approved_ids) - latest_snapshot)
        gpu = gpu_status()
        minimum = int(settings["minimum_approved_samples"])
        return {
            "settings": settings,
            "readiness": {
                "approved_samples": len(approved_ids),
                "minimum_approved_samples": minimum,
                "new_samples_since_last_job": new_count,
                "minimum_new_samples": int(settings["minimum_new_samples"]),
                "gpu": gpu,
                "ready": len(approved_ids) >= minimum and bool(gpu.get("available")) and active_job(self.private_root) is None,
                "safety_policy": "实拍测试集永久隔离；候选模型只有通过成对质量门禁才允许替换线上权重。",
            },
            "active_model": active_model_manifest(),
            "active_deployment": active_deployment(self.private_root),
            "active_job": active_job(self.private_root),
            "jobs": jobs,
        }

    def update_training_settings(self, principal: dict[str, Any], payload: Any) -> dict[str, Any]:
        self.require_owner(principal)
        if not isinstance(payload, dict):
            raise PlatformError(422, "INVALID_TRAINING_SETTINGS", "自动训练设置格式不正确。")
        allowed = {
            "enabled",
            "minimum_approved_samples",
            "minimum_new_samples",
            "epochs",
            "history_limit",
            "offline_optimization",
            "optimization_trials",
            "screening_epochs",
            "hard_example_replay",
            "targets",
            "auto_promote",
            "require_quality_gate",
        }
        if set(payload) != allowed:
            raise PlatformError(422, "INVALID_TRAINING_SETTINGS", "自动训练设置字段不完整或包含未知字段。")
        from .auto_training import VALID_TARGETS, save_settings

        for key in (
            "enabled", "auto_promote", "require_quality_gate", "offline_optimization", "hard_example_replay"
        ):
            if not isinstance(payload[key], bool):
                raise PlatformError(422, "INVALID_TRAINING_SETTINGS", f"{key} 必须是开关值。")
        numeric_limits = {
            "minimum_approved_samples": (20, 500),
            "minimum_new_samples": (5, 200),
            "epochs": (1, 100),
            "history_limit": (0, 500),
            "optimization_trials": (1, 3),
            "screening_epochs": (1, 20),
        }
        normalized = dict(payload)
        for key, (minimum, maximum) in numeric_limits.items():
            value = payload[key]
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise PlatformError(
                    422,
                    "INVALID_TRAINING_SETTINGS",
                    f"{key} 必须是 {minimum} 到 {maximum} 之间的整数。",
                )
        targets = payload["targets"]
        if (
            not isinstance(targets, list)
            or not targets
            or any(not isinstance(value, str) for value in targets)
            or len(set(targets)) != len(targets)
            or set(targets) - VALID_TARGETS
        ):
            raise PlatformError(422, "INVALID_TRAINING_SETTINGS", "训练目标不正确或存在重复。")
        if payload["auto_promote"] and not payload["require_quality_gate"]:
            raise PlatformError(422, "QUALITY_GATE_REQUIRED", "自动更新线上模型时不能关闭质量门禁。")
        if payload["offline_optimization"] and payload["optimization_trials"] < 2:
            raise PlatformError(422, "OFFLINE_SEARCH_TRIALS_REQUIRED", "离线智能优化至少需要 2 个候选方案。")
        normalized["targets"] = list(targets)
        return {"settings": save_settings(self.private_root, normalized)}

    def _create_auto_training_job(self, *, trigger: str, ignore_new_threshold: bool = False) -> dict[str, Any]:
        from .auto_training import active_job, create_job, gpu_status, list_jobs, load_settings

        settings = load_settings(self.private_root)
        approved_ids = self._approved_training_feedback_ids()
        if len(approved_ids) < int(settings["minimum_approved_samples"]):
            raise PlatformError(
                409,
                "INSUFFICIENT_APPROVED_SAMPLES",
                "审核通过的实拍标注还不足，系统不会启动容易过拟合的训练。",
                details={
                    "approved": len(approved_ids),
                    "minimum": int(settings["minimum_approved_samples"]),
                },
            )
        if active_job(self.private_root) is not None:
            raise PlatformError(409, "TRAINING_JOB_ACTIVE", "已有自动训练任务正在运行。")
        gpu = gpu_status()
        if not gpu.get("available"):
            raise PlatformError(409, "TRAINING_GPU_UNAVAILABLE", "没有检测到可用于训练的 NVIDIA CUDA 显卡。")
        if not ignore_new_threshold:
            jobs = list_jobs(self.private_root, 100)
            latest_snapshot = max((int(job.get("approved_snapshot_count", 0)) for job in jobs), default=0)
            new_count = max(0, len(approved_ids) - latest_snapshot)
            if new_count < int(settings["minimum_new_samples"]):
                raise PlatformError(
                    409,
                    "INSUFFICIENT_NEW_SAMPLES",
                    "距离上次训练新增的实拍标注还不足。",
                    details={"new_samples": new_count, "minimum": int(settings["minimum_new_samples"])},
                )
        try:
            return create_job(self.private_root, approved_ids, settings, trigger=trigger)
        except RuntimeError as exc:
            raise PlatformError(409, "TRAINING_JOB_ACTIVE", str(exc)) from exc
        except OSError as exc:
            raise PlatformError(500, "TRAINING_JOB_START_FAILED", f"自动训练任务创建失败：{exc}") from exc

    def start_auto_training(self, principal: dict[str, Any], payload: Any) -> dict[str, Any]:
        self.require_owner(principal)
        if not isinstance(payload, dict) or set(payload) != {"confirm"} or payload.get("confirm") is not True:
            raise PlatformError(422, "TRAINING_CONFIRMATION_REQUIRED", "请明确确认启动自动训练。")
        with self._auto_training_lock:
            job = self._create_auto_training_job(trigger="manual", ignore_new_threshold=True)
        return {"training_job": job}

    def _maybe_schedule_auto_training(self) -> dict[str, Any] | None:
        from .auto_training import load_settings

        settings = load_settings(self.private_root)
        if not settings.get("enabled"):
            return None
        if not self._auto_training_lock.acquire(blocking=False):
            return None
        try:
            try:
                return self._create_auto_training_job(trigger="approved_feedback", ignore_new_threshold=False)
            except PlatformError:
                return None
        finally:
            self._auto_training_lock.release()

    def rollback_auto_model(self, principal: dict[str, Any], payload: Any) -> dict[str, Any]:
        self.require_owner(principal)
        if not isinstance(payload, dict) or set(payload) != {"confirm"} or payload.get("confirm") is not True:
            raise PlatformError(422, "ROLLBACK_CONFIRMATION_REQUIRED", "请明确确认回滚模型。")
        from .auto_training import active_job, rollback_latest

        if active_job(self.private_root) is not None:
            raise PlatformError(409, "TRAINING_JOB_ACTIVE", "训练任务运行时不能回滚模型。")
        try:
            deployment = rollback_latest(self.private_root)
        except RuntimeError as exc:
            raise PlatformError(409, "MODEL_ROLLBACK_UNAVAILABLE", str(exc)) from exc
        except OSError as exc:
            raise PlatformError(500, "MODEL_ROLLBACK_FAILED", f"模型回滚失败：{exc}") from exc
        return {"deployment": deployment, "active_model": self.admin_training_status(principal)["active_model"]}

    def _public_tenant(self, tenant: dict[str, Any], token: str | None) -> dict[str, Any]:
        item = copy.deepcopy(tenant)
        expiry = self._parse_utc(item.get("expires_at"))
        item["expired"] = bool(expiry and expiry <= datetime.now(timezone.utc))
        item["share_url"] = self._share_url(token)
        item["login_url"] = self._login_url()
        return item

    def _set_tenant_credentials(
        self,
        tenant_id: str,
        username_value: Any,
        password_value: Any = None,
    ) -> tuple[dict[str, Any], str]:
        username = self._normalize_username(username_value or self._unique_username())
        password = self._validate_password(password_value or _generated_password(), username)
        existing = self.database.tenant_auth_record(username)
        if existing and str(existing["id"]) != tenant_id:
            raise PlatformError(409, "USERNAME_EXISTS", "该登录账号已被其他企业使用。")
        if self.database.admin_user_auth_record(username):
            raise PlatformError(409, "USERNAME_EXISTS", "该登录账号已被管理员使用。")
        try:
            tenant = self.database.set_tenant_credentials(tenant_id, username, _password_hash(password))
        except sqlite3.IntegrityError as exc:
            raise PlatformError(409, "USERNAME_EXISTS", "该登录账号已被其他企业使用。") from exc
        return tenant, password

    def admin_users(self, principal: dict[str, Any]) -> dict[str, Any]:
        self.require_owner(principal)
        return {
            "admins": self.database.list_admin_users(),
            "login_url": self._admin_login_url(),
        }

    def create_admin_user(self, principal: dict[str, Any], payload: Any) -> dict[str, Any]:
        self.require_owner(principal)
        if not isinstance(payload, dict):
            raise PlatformError(422, "INVALID_ADMIN_USER", "管理员信息格式不正确。")
        label = str(payload.get("label") or "").strip()
        if not label or len(label) > 80:
            raise PlatformError(422, "INVALID_ADMIN_LABEL", "管理员姓名需为 1 到 80 个字符。")
        username = self._normalize_username(payload.get("username") or self._unique_admin_username())
        password = self._validate_password(payload.get("password") or _generated_password(), username)
        if self.database.tenant_auth_record(username) or self.database.admin_user_auth_record(username):
            raise PlatformError(409, "USERNAME_EXISTS", "该登录账号已被使用。")
        try:
            admin = self.database.create_admin_user(
                label, username, _password_hash(password), access_level="annotator"
            )
        except sqlite3.IntegrityError as exc:
            raise PlatformError(409, "USERNAME_EXISTS", "该登录账号已被使用。") from exc
        return {
            "admin": admin,
            "credentials": {
                "label": label,
                "username": username,
                "initial_password": password,
                "login_url": self._admin_login_url(),
            },
        }

    def admin_user_action(
        self, principal: dict[str, Any], admin_user_id: str, payload: Any
    ) -> dict[str, Any]:
        self.require_owner(principal)
        admin = self.database.admin_user(admin_user_id)
        if admin is None or not SAFE_ID.fullmatch(admin_user_id):
            raise PlatformError(404, "ADMIN_USER_NOT_FOUND", "标注管理员不存在。")
        if admin.get("access_level") != "annotator":
            raise PlatformError(403, "OWNER_ACCOUNT_PROTECTED", "平台主管理员账号不能在这里修改。")
        if not isinstance(payload, dict):
            raise PlatformError(422, "INVALID_ADMIN_ACTION", "管理员操作格式不正确。")
        action = str(payload.get("action") or "").strip()
        credentials = None
        if action in {"enable", "disable"}:
            admin = self.database.set_admin_user_active(admin_user_id, action == "enable")
        elif action == "reset_credentials":
            username = self._normalize_username(payload.get("username") or admin.get("username"))
            password = self._validate_password(payload.get("password") or _generated_password(), username)
            tenant_record = self.database.tenant_auth_record(username)
            other_admin = self.database.admin_user_auth_record(username)
            if tenant_record or (other_admin and str(other_admin["id"]) != admin_user_id):
                raise PlatformError(409, "USERNAME_EXISTS", "该登录账号已被使用。")
            try:
                admin = self.database.set_admin_user_credentials(
                    admin_user_id, username, _password_hash(password)
                )
            except sqlite3.IntegrityError as exc:
                raise PlatformError(409, "USERNAME_EXISTS", "该登录账号已被使用。") from exc
            credentials = {
                "label": admin["label"],
                "username": username,
                "initial_password": password,
                "login_url": self._admin_login_url(),
            }
        else:
            raise PlatformError(422, "INVALID_ADMIN_ACTION", "不支持此管理员操作。")
        result = {"admin": admin}
        if credentials:
            result["credentials"] = credentials
        return result

    def delete_admin_user(
        self, principal: dict[str, Any], admin_user_id: str, payload: Any
    ) -> dict[str, Any]:
        self.require_owner(principal)
        admin = self.database.admin_user(admin_user_id)
        if admin is None or not SAFE_ID.fullmatch(admin_user_id):
            raise PlatformError(404, "ADMIN_USER_NOT_FOUND", "标注管理员不存在。")
        if admin.get("access_level") != "annotator":
            raise PlatformError(403, "OWNER_ACCOUNT_PROTECTED", "平台主管理员账号不能删除。")
        if not isinstance(payload, dict) or str(payload.get("confirm_username") or "") != admin["username"]:
            raise PlatformError(422, "ADMIN_DELETE_CONFIRMATION_REQUIRED", "请输入完整登录账号确认删除。")
        if not self.database.delete_admin_user(admin_user_id):
            raise PlatformError(404, "ADMIN_USER_NOT_FOUND", "标注管理员不存在。")
        return {"deleted": True, "admin": {"id": admin_user_id, "username": admin["username"]}}

    def admin_tenants(self, principal: dict[str, Any]) -> dict[str, Any]:
        self.require_owner(principal)
        access = self._read_access()
        tokens = access.get("enterprise_tokens")
        if not isinstance(tokens, dict):
            tokens = {}
        tenants = [
            self._public_tenant(tenant, str(tokens.get(tenant["id"]) or "") or None)
            for tenant in self.database.list_tenants()
        ]
        return {"tenants": tenants, "summary": self.database.summary()}

    def create_tenant(self, principal: dict[str, Any], payload: Any) -> dict[str, Any]:
        self.require_owner(principal)
        if not isinstance(payload, dict):
            raise PlatformError(422, "INVALID_TENANT", "企业信息格式不正确。")
        name = str(payload.get("name") or "").strip()
        if not name or len(name) > 120:
            raise PlatformError(422, "INVALID_TENANT_NAME", "企业名称需为 1 到 120 个字符。")
        try:
            valid_days = int(payload.get("valid_days", 365))
        except (TypeError, ValueError) as exc:
            raise PlatformError(422, "INVALID_VALID_DAYS", "有效期天数必须是整数。") from exc
        if not 1 <= valid_days <= 3650:
            raise PlatformError(422, "INVALID_VALID_DAYS", "有效期需设置为 1 到 3650 天。")
        requested_username = self._normalize_username(payload.get("username") or self._unique_username())
        requested_password = self._validate_password(payload.get("password") or _generated_password(), requested_username)
        project = self.store.create_project(
            {
                "name": f"企业回传检测 - {name}",
                "description": f"CardScope 企业端独立检测空间：{name}",
                "default_classification": {
                    "face": "front",
                    "layout_id": "gx_current",
                    "orientation_degrees_cw": 0,
                    "card_type": "",
                },
                "custom_metadata": {"platform_managed": True, "tenant_name": name},
            }
        )
        token = "ent_" + secrets.token_urlsafe(32)
        tenant = self.database.create_tenant(
            name,
            project["id"],
            token,
            expires_at=self._future_expiry(valid_days),
        )
        tenant, initial_password = self._set_tenant_credentials(
            tenant["id"], requested_username, requested_password
        )
        access = self._read_access()
        tokens = access.setdefault("enterprise_tokens", {})
        if not isinstance(tokens, dict):
            tokens = {}
            access["enterprise_tokens"] = tokens
        tokens[tenant["id"]] = token
        self._write_access(access)
        public_tenant = self._public_tenant(tenant, token)
        return {
            "tenant": public_tenant,
            "credentials": {
                "username": public_tenant["username"],
                "initial_password": initial_password,
                "login_url": public_tenant["login_url"],
            },
        }

    def tenant_action(
        self,
        principal: dict[str, Any],
        tenant_id: str,
        payload: Any,
    ) -> dict[str, Any]:
        self.require_owner(principal)
        if not SAFE_ID.fullmatch(tenant_id) or self.database.tenant(tenant_id) is None:
            raise PlatformError(404, "TENANT_NOT_FOUND", "企业不存在。")
        if not isinstance(payload, dict):
            raise PlatformError(422, "INVALID_TENANT_ACTION", "企业操作格式不正确。")
        action = str(payload.get("action") or "").strip()
        access = self._read_access()
        tokens = access.setdefault("enterprise_tokens", {})
        if not isinstance(tokens, dict):
            tokens = {}
            access["enterprise_tokens"] = tokens
        if action == "rotate":
            token = "ent_" + secrets.token_urlsafe(32)
            tenant = self.database.rotate_tenant_token(tenant_id, token)
            tokens[tenant_id] = token
            if str(access.get("tenant_id") or "") == tenant_id:
                access["enterprise_token"] = token
                access["enterprise_url"] = self._login_url()
                access["legacy_enterprise_url"] = self._share_url(token)
            self._write_access(access)
        elif action in {"enable", "disable"}:
            tenant = self.database.set_tenant_active(tenant_id, action == "enable")
        elif action == "extend":
            try:
                days = int(payload.get("days", 365))
            except (TypeError, ValueError) as exc:
                raise PlatformError(422, "INVALID_VALID_DAYS", "延期天数必须是整数。") from exc
            if not 1 <= days <= 3650:
                raise PlatformError(422, "INVALID_VALID_DAYS", "延期天数需为 1 到 3650 天。")
            current = self.database.tenant(tenant_id)
            assert current is not None
            tenant = self.database.set_tenant_expiry(
                tenant_id,
                self._future_expiry(days, current.get("expires_at")),
            )
        elif action == "set_credentials":
            current = self.database.tenant(tenant_id)
            assert current is not None
            username = payload.get("username") or current.get("username") or self._unique_username()
            tenant, initial_password = self._set_tenant_credentials(
                tenant_id, username, payload.get("password")
            )
            return {
                "tenant": self._public_tenant(tenant, str(tokens.get(tenant_id) or "") or None),
                "credentials": {
                    "username": tenant["username"],
                    "initial_password": initial_password,
                    "login_url": self._login_url(),
                },
            }
        else:
            raise PlatformError(422, "INVALID_TENANT_ACTION", "不支持此企业操作。")
        return {"tenant": self._public_tenant(tenant, str(tokens.get(tenant_id) or "") or None)}

    def delete_tenant(
        self,
        principal: dict[str, Any],
        tenant_id: str,
        payload: Any,
    ) -> dict[str, Any]:
        self.require_owner(principal)
        self._require_deletion_idle()
        manifest = self.database.tenant_deletion_manifest(tenant_id)
        tenant_batch_ids = self.database.batch_ids_for_tenant(tenant_id)
        if manifest is None or not SAFE_ID.fullmatch(tenant_id):
            raise PlatformError(404, "TENANT_NOT_FOUND", "企业不存在。")
        if not isinstance(payload, dict) or str(payload.get("confirm_name") or "") != manifest["name"]:
            raise PlatformError(422, "TENANT_DELETE_CONFIRMATION_REQUIRED", "请输入完整企业名称确认删除。")
        access = self._read_access()
        tokens = access.get("enterprise_tokens")
        if not isinstance(tokens, dict):
            tokens = {}
        tokens.pop(tenant_id, None)
        access["enterprise_tokens"] = tokens
        if str(access.get("tenant_id") or "") == tenant_id:
            access.pop("tenant_id", None)
            access.pop("enterprise_token", None)
            access["enterprise_url"] = self._login_url()
            access["legacy_enterprise_url"] = None
        counts = self.database.delete_tenant(tenant_id)
        self._write_access(access)
        cleanup_warning = None
        for batch_id in tenant_batch_ids:
            batch_root = (self.batch_spool_root / batch_id).resolve()
            try:
                batch_root.relative_to(self.batch_spool_root.resolve())
                shutil.rmtree(batch_root, ignore_errors=True)
            except ValueError:
                cleanup_warning = "企业记录已删除，但批量上传缓存路径校验失败。"
        try:
            project_cleanup = self.store.delete_project(str(manifest["project_id"]))
        except (StudioError, OSError) as exc:
            project_cleanup = {"deleted": False, "project_id": manifest["project_id"]}
            cleanup_warning = f"企业记录已删除，但残留文件清理失败：{exc}"
        return {
            "deleted": True,
            "tenant": {"id": tenant_id, "name": manifest["name"]},
            "deleted_counts": counts,
            "project_cleanup": project_cleanup,
            "cleanup_warning": cleanup_warning,
            "summary": self.database.summary(),
        }

    @staticmethod
    def _validate_batch_manifest(payload: Any) -> tuple[list[dict[str, Any]], int]:
        if not isinstance(payload, dict) or set(payload) != {"items"}:
            raise PlatformError(
                422,
                "INVALID_BATCH_MANIFEST",
                "批量任务必须包含完整的图片清单。",
            )
        raw_items = payload.get("items")
        if not isinstance(raw_items, list) or not 1 <= len(raw_items) <= MAX_BATCH_FILES:
            raise PlatformError(
                422,
                "INVALID_BATCH_SIZE",
                f"一次请选择 1 至 {MAX_BATCH_FILES} 张图片。",
            )
        items: list[dict[str, Any]] = []
        client_keys: set[str] = set()
        total_bytes = 0
        for index, raw in enumerate(raw_items):
            if not isinstance(raw, dict):
                raise PlatformError(422, "INVALID_BATCH_ITEM", "图片清单格式不正确。")
            client_key = str(raw.get("client_key") or "").strip()
            if not client_key or len(client_key) > 300 or client_key in client_keys:
                raise PlatformError(
                    422,
                    "INVALID_BATCH_ITEM_KEY",
                    "图片清单包含重复或无效的文件标识。",
                )
            filename = str(raw.get("filename") or "").replace("\\", "/").split("/")[-1].strip()
            if not filename or len(filename) > 255 or "\x00" in filename:
                raise PlatformError(422, "INVALID_BATCH_FILENAME", "图片文件名无效。")
            content_type = str(raw.get("content_type") or "application/octet-stream").lower()
            if content_type not in BATCH_CONTENT_TYPES:
                raise PlatformError(
                    415,
                    "IMAGE_REQUIRED",
                    f"{filename} 不是支持的 JPG、PNG 或 WebP 图片。",
                )
            raw_size = raw.get("size")
            if isinstance(raw_size, bool):
                raise PlatformError(422, "INVALID_BATCH_FILE_SIZE", "图片大小格式不正确。")
            try:
                size = int(raw_size)
            except (TypeError, ValueError) as exc:
                raise PlatformError(
                    422, "INVALID_BATCH_FILE_SIZE", "图片大小格式不正确。"
                ) from exc
            if size <= 0 or size > MAX_BATCH_FILE_BYTES:
                raise PlatformError(
                    413,
                    "BATCH_FILE_TOO_LARGE",
                    f"{filename} 超过单张 100 MB 限制或文件为空。",
                )
            total_bytes += size
            if total_bytes > MAX_BATCH_TOTAL_BYTES:
                raise PlatformError(
                    413,
                    "BATCH_TOO_LARGE",
                    "本批图片总大小超过 5 GB，请拆分后上传。",
                )
            client_keys.add(client_key)
            items.append(
                {
                    "position": index,
                    "client_key": client_key,
                    "filename": filename,
                    "content_type": content_type,
                    "size": size,
                }
            )
        return items, total_bytes

    def _authorized_batch(
        self, principal: dict[str, Any], batch_id: str
    ) -> dict[str, Any]:
        self.require_role(principal, "enterprise")
        if not SAFE_ID.fullmatch(batch_id):
            raise PlatformError(404, "BATCH_NOT_FOUND", "批量任务不存在。")
        batch = self.database.batch(batch_id)
        if (
            batch is None
            or str(batch["tenant_id"]) != str(principal["tenant"]["id"])
        ):
            raise PlatformError(404, "BATCH_NOT_FOUND", "批量任务不存在。")
        return batch

    @staticmethod
    def _public_batch(batch: dict[str, Any]) -> dict[str, Any]:
        public = copy.deepcopy(batch)
        for item in public.get("items", []):
            item.pop("spool_path", None)
            item.pop("source_sha256", None)
        counts = public.get("counts") or {}
        public["uploaded_count"] = (
            int(public.get("expected_count", 0)) - int(counts.get("waiting_upload", 0))
        )
        public["processed_count"] = (
            int(counts.get("completed", 0))
            + int(counts.get("failed", 0))
            + int(counts.get("cancelled", 0))
        )
        return public

    def create_batch_job(
        self, principal: dict[str, Any], payload: Any
    ) -> dict[str, Any]:
        self.require_role(principal, "enterprise")
        items, total_bytes = self._validate_batch_manifest(payload)
        batch_id = self.database.create_batch(
            str(principal["tenant"]["id"]), items, total_bytes
        )
        batch = self.database.batch(batch_id)
        if batch is None:
            raise PlatformError(500, "BATCH_CREATE_FAILED", "批量任务创建失败。")
        return self._public_batch(batch)

    def list_batch_jobs(
        self, principal: dict[str, Any], limit: int = 20
    ) -> list[dict[str, Any]]:
        self.require_role(principal, "enterprise")
        return [
            self._public_batch(batch)
            for batch in self.database.list_batches(
                str(principal["tenant"]["id"]), limit=limit
            )
        ]

    def batch_job(
        self,
        principal: dict[str, Any],
        batch_id: str,
        *,
        include_inspections: bool = False,
    ) -> dict[str, Any]:
        batch = self._authorized_batch(principal, batch_id)
        public = self._public_batch(batch)
        if include_inspections:
            for item in public["items"]:
                inspection_id = item.get("inspection_id")
                item["inspection"] = (
                    self.inspection(principal, str(inspection_id))
                    if inspection_id
                    else None
                )
        return public

    def _batch_spool_path(self, batch_id: str, item_id: str) -> Path:
        if not SAFE_ID.fullmatch(batch_id) or not SAFE_ID.fullmatch(item_id):
            raise PlatformError(404, "BATCH_ITEM_NOT_FOUND", "批量图片不存在。")
        batch_root = (self.batch_spool_root / batch_id).resolve()
        try:
            batch_root.relative_to(self.batch_spool_root.resolve())
        except ValueError as exc:
            raise PlatformError(500, "BATCH_SPOOL_INVALID", "批量缓存路径无效。") from exc
        batch_root.mkdir(parents=True, exist_ok=True)
        return batch_root / f"{item_id}.upload"

    def _resolve_spool_path(self, relative: str) -> Path:
        path = (self.private_root / relative).resolve()
        try:
            path.relative_to(self.batch_spool_root.resolve())
        except ValueError as exc:
            raise PlatformError(500, "BATCH_SPOOL_INVALID", "批量缓存路径无效。") from exc
        return path

    def upload_batch_item(
        self,
        principal: dict[str, Any],
        batch_id: str,
        item_id: str,
        payload: bytes,
        content_type: str,
    ) -> dict[str, Any]:
        batch = self._authorized_batch(principal, batch_id)
        if str(batch["state"]) in {"cancelled", "completed"}:
            raise PlatformError(409, "BATCH_NOT_UPLOADABLE", "该批量任务已结束。")
        item = self.database.batch_item(batch_id, item_id)
        if item is None:
            raise PlatformError(404, "BATCH_ITEM_NOT_FOUND", "批量图片不存在。")
        if str(item["state"]) != "waiting_upload":
            return {
                "item": self._public_batch({"items": [item], "counts": {}})["items"][0],
                "batch": self.batch_job(principal, batch_id),
            }
        normalized_type = content_type.split(";", 1)[0].strip().lower()
        if normalized_type not in BATCH_CONTENT_TYPES:
            raise PlatformError(415, "IMAGE_REQUIRED", "请上传 JPG、PNG 或 WebP 图片。")
        if not payload:
            raise PlatformError(400, "EMPTY_UPLOAD", "图片文件为空。")
        if len(payload) != int(item["expected_bytes"]):
            raise PlatformError(
                409,
                "BATCH_FILE_CHANGED",
                f"{item['filename']} 的大小与创建任务时不同，请重新选择原文件。",
            )
        if len(payload) > MAX_BATCH_FILE_BYTES:
            raise PlatformError(413, "BATCH_FILE_TOO_LARGE", "单张图片不能超过 100 MB。")
        with self._batch_upload_lock:
            current = self.database.batch_item(batch_id, item_id)
            if current is None:
                raise PlatformError(404, "BATCH_ITEM_NOT_FOUND", "批量图片不存在。")
            if str(current["state"]) != "waiting_upload":
                return {
                    "item": self._public_batch(
                        {"items": [current], "counts": {}}
                    )["items"][0],
                    "batch": self.batch_job(principal, batch_id),
                }
            target = self._batch_spool_path(batch_id, item_id)
            temporary = target.with_suffix(f".tmp-{secrets.token_hex(6)}")
            try:
                temporary.write_bytes(payload)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
            relative = target.relative_to(self.private_root).as_posix()
            item = self.database.mark_batch_item_uploaded(
                batch_id,
                item_id,
                spool_path=relative,
                source_sha256=hashlib.sha256(payload).hexdigest(),
            )
        self.start_batch_worker()
        self._batch_worker_wake.set()
        return {
            "item": self._public_batch({"items": [item], "counts": {}})["items"][0],
            "batch": self.batch_job(principal, batch_id),
        }

    def batch_job_action(
        self, principal: dict[str, Any], batch_id: str, payload: Any
    ) -> dict[str, Any]:
        batch = self._authorized_batch(principal, batch_id)
        if not isinstance(payload, dict) or set(payload) != {"action"}:
            raise PlatformError(422, "INVALID_BATCH_ACTION", "批量任务操作格式不正确。")
        action = str(payload.get("action") or "")
        if action not in {"pause", "resume", "retry", "cancel"}:
            raise PlatformError(422, "INVALID_BATCH_ACTION", "不支持该批量任务操作。")
        self.database.batch_action(batch_id, action)
        if action == "cancel":
            refreshed = self.database.batch(batch_id) or batch
            for item in refreshed.get("items", []):
                relative = item.get("spool_path")
                if relative and str(item.get("state")) == "cancelled":
                    self._resolve_spool_path(str(relative)).unlink(missing_ok=True)
        elif action in {"resume", "retry"}:
            self.start_batch_worker()
            self._batch_worker_wake.set()
        return self.batch_job(principal, batch_id)

    def start_batch_worker(self) -> None:
        with self._batch_worker_lock:
            if self._batch_worker is not None and self._batch_worker.is_alive():
                return
            self._batch_worker_stop.clear()
            self.database.recover_batch_queue()
            self._batch_worker = threading.Thread(
                target=self._batch_worker_loop,
                name="cardscope-batch-worker",
                daemon=True,
            )
            self._batch_worker.start()
            self._batch_worker_wake.set()

    def stop_batch_worker(self, timeout: float = 5.0) -> None:
        self._batch_worker_stop.set()
        self._batch_worker_wake.set()
        worker = self._batch_worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=max(0.0, timeout))

    def _batch_worker_loop(self) -> None:
        while not self._batch_worker_stop.is_set():
            item = self.database.claim_next_batch_item()
            if item is None:
                self._batch_worker_wake.wait(timeout=2.0)
                self._batch_worker_wake.clear()
                continue
            self._process_batch_item(item)

    def _process_batch_item(self, item: dict[str, Any]) -> None:
        batch_id = str(item["batch_id"])
        item_id = str(item["id"])
        relative = str(item.get("spool_path") or "")
        try:
            if not relative:
                raise PlatformError(500, "BATCH_FILE_MISSING", "批量缓存文件不存在。")
            source = self._resolve_spool_path(relative)
            if not source.is_file():
                raise PlatformError(500, "BATCH_FILE_MISSING", "批量缓存文件不存在。")
            payload = source.read_bytes()
            expected_hash = str(item.get("source_sha256") or "")
            if expected_hash and hashlib.sha256(payload).hexdigest() != expected_hash:
                raise PlatformError(500, "BATCH_FILE_CORRUPT", "批量缓存文件校验失败。")
            tenant = self.database.tenant(str(item["tenant_id"]))
            if tenant is None:
                raise PlatformError(404, "TENANT_NOT_FOUND", "企业账号不存在。")
            principal = {"role": "enterprise", "tenant": tenant}
            inspection = self.create_inspection(
                principal, str(item["filename"]), payload
            )
            self.database.finish_batch_item(
                batch_id, item_id, inspection_id=str(inspection["id"])
            )
            source.unlink(missing_ok=True)
            try:
                source.parent.rmdir()
            except OSError:
                pass
        except PlatformError as exc:
            self.database.finish_batch_item(
                batch_id,
                item_id,
                error_code=exc.code,
                error_message=exc.message,
            )
        except Exception as exc:
            self.database.finish_batch_item(
                batch_id,
                item_id,
                error_code="BATCH_PROCESSING_FAILED",
                error_message=f"后台检测失败：{exc}",
            )

    def create_inspection(
        self,
        principal: dict[str, Any],
        filename: str,
        payload: bytes,
    ) -> dict[str, Any]:
        self.require_role(principal, "enterprise")
        tenant = principal["tenant"]
        if not payload:
            raise PlatformError(400, "EMPTY_UPLOAD", "请选择一张卡牌图片。")
        batch = self.store.create_import(
            tenant["project_id"], {"expected_files": 1, "source": "enterprise-platform"}
        )
        try:
            imported = self.store.import_file(
                tenant["project_id"], batch["id"], filename, payload
            )
            sample = imported["sample"]
            prelabel = self.store.generate_prelabel(
                tenant["project_id"], sample["id"], "gx_current", force_recompute=True
            )
        except StudioError as exc:
            raise PlatformError(exc.status, exc.code, exc.message, exc.details) from exc
        prediction = _compact_prediction(prelabel, self.pass_deviation_percent)
        prediction["source_size"] = {
            "width": int(sample["width"]),
            "height": int(sample["height"]),
        }
        has_complete_result = bool(
            prediction.get("outer_corners") and prediction.get("inner_lines_rectified")
        )
        state = "completed" if has_complete_result else "detection_failed"
        inspection_id = self.database.add_inspection(
            tenant_id=tenant["id"],
            project_id=tenant["project_id"],
            sample_id=sample["id"],
            filename=sample["filename"],
            state=state,
            model_version=prediction["model_version"],
            prediction=prediction,
        )
        inspection = self.inspection(principal, inspection_id)
        write_inspection_result(self.workspace, inspection)
        return inspection

    def create_reference_job(self, principal: dict[str, Any], payload: Any) -> dict[str, Any]:
        self.require_role(principal, "enterprise")
        if not isinstance(payload, dict):
            raise PlatformError(422, "INVALID_REFERENCE_JOB", "参考图任务格式不正确。")
        capture = payload.get("capture") if isinstance(payload.get("capture"), dict) else {}
        reference = payload.get("reference") if isinstance(payload.get("reference"), dict) else {}
        capture_name = Path(str(capture.get("filename") or "")).name
        reference_name = Path(str(reference.get("filename") or "")).name
        if not capture_name or not reference_name:
            raise PlatformError(422, "REFERENCE_FILENAMES_REQUIRED", "请分别选择实拍图和标准图。")
        tenant = principal["tenant"]
        job_id = self.database.create_reference_job(tenant["id"], tenant["project_id"], capture_name, reference_name)
        return self.reference_job(principal, job_id)

    def reference_job(self, principal: dict[str, Any], job_id: str) -> dict[str, Any]:
        job = self.database.reference_job(job_id)
        if job is None or (principal["role"] == "enterprise" and job["tenant_id"] != principal["tenant"]["id"]):
            raise PlatformError(404, "REFERENCE_JOB_NOT_FOUND", "参考图任务不存在。")
        return {key: value for key, value in job.items() if key not in {"capture_path", "reference_path"}}

    def upload_reference_file(
        self, principal: dict[str, Any], job_id: str, kind: str, payload: bytes, content_type: str
    ) -> dict[str, Any]:
        self.require_role(principal, "enterprise")
        if kind not in {"capture", "reference"}:
            raise PlatformError(404, "REFERENCE_FILE_NOT_FOUND", "参考图文件类型无效。")
        if content_type.split(";", 1)[0].strip().lower() not in BATCH_CONTENT_TYPES:
            raise PlatformError(415, "IMAGE_REQUIRED", "请上传 JPG、PNG 或 WebP 图片。")
        if not payload or len(payload) > MAX_BATCH_FILE_BYTES:
            raise PlatformError(413, "REFERENCE_FILE_INVALID", "参考图文件为空或过大。")
        job = self.database.reference_job(job_id)
        if job is None or job["tenant_id"] != principal["tenant"]["id"]:
            raise PlatformError(404, "REFERENCE_JOB_NOT_FOUND", "参考图任务不存在。")
        target = (self.reference_spool_root / job_id / f"{kind}.upload").resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        job = self.database.mark_reference_file_uploaded(job_id, kind, str(target.relative_to(self.private_root)))
        if job["state"] != "ready":
            return {"job": self.reference_job(principal, job_id)}
        return {"job": self.reference_job(principal, job_id), "inspection": self._process_reference_job(principal, job)}

    def _process_reference_job(self, principal: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
        capture_path = self.private_root / str(job["capture_path"])
        reference_path = self.private_root / str(job["reference_path"])
        try:
            prediction, assets = run_reference_registration(
                capture_png=capture_path.read_bytes(), reference_png=reference_path.read_bytes()
            )
            batch = self.store.create_import(principal["tenant"]["project_id"], {"expected_files": 1, "source": "reference-registration"})
            imported = self.store.import_file(principal["tenant"]["project_id"], batch["id"], str(job["capture_filename"]), capture_path.read_bytes())
        except (OSError, StudioError, RegistrationInputError) as exc:
            raise PlatformError(422, getattr(exc, "code", "REFERENCE_REGISTRATION_FAILED"), str(exc)) from exc
        sample = imported["sample"]
        prediction["source_size"] = {"width": int(sample["width"]), "height": int(sample["height"])}
        state = "completed" if prediction.get("success") else "detection_failed"
        inspection_id = self.database.add_inspection(
            tenant_id=principal["tenant"]["id"], project_id=principal["tenant"]["project_id"],
            sample_id=sample["id"], filename=sample["filename"], state=state,
            model_version=prediction.get("model_version"), prediction=prediction,
        )
        asset_root = self.private_root / "reference_assets" / inspection_id
        asset_root.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reference_path, asset_root / "reference.upload")
        self.database.finish_reference_job(str(job["id"]), inspection_id)
        inspection = self.inspection(principal, inspection_id)
        write_inspection_result(self.workspace, inspection)
        return inspection

    def _authorize_inspection(
        self, principal: dict[str, Any], inspection_id: str
    ) -> dict[str, Any]:
        if not SAFE_ID.fullmatch(inspection_id):
            raise PlatformError(404, "INSPECTION_NOT_FOUND", "检测记录不存在。")
        inspection = self.database.inspection(inspection_id)
        if inspection is None:
            raise PlatformError(404, "INSPECTION_NOT_FOUND", "检测记录不存在。")
        if principal["role"] == "enterprise" and inspection["tenant_id"] != principal["tenant"]["id"]:
            raise PlatformError(404, "INSPECTION_NOT_FOUND", "检测记录不存在。")
        if (
            principal["role"] == "admin"
            and self.admin_access_level(principal) == "annotator"
            and self.database.feedback_for_inspection(inspection_id) is None
        ):
            raise PlatformError(404, "INSPECTION_NOT_FOUND", "该检测记录不在人工标注任务中。")
        return inspection

    def inspection(self, principal: dict[str, Any], inspection_id: str) -> dict[str, Any]:
        item = self._authorize_inspection(principal, inspection_id)
        public = copy.deepcopy(item)
        public["images"] = {
            "preview": f"/api/platform/v1/inspections/{item['id']}/image?variant=preview",
            "rectified": f"/api/platform/v1/inspections/{item['id']}/image?variant=rectified",
            "display_preview": (
                f"/api/platform/v1/inspections/{item['id']}/image"
                "?variant=preview&display=webp"
            ),
            "display_rectified": (
                f"/api/platform/v1/inspections/{item['id']}/image"
                "?variant=rectified&display=webp"
            ),
        }
        if public.get("prediction", {}).get("measurement_mode") == "reference_registration":
            public["images"]["reference"] = f"/api/platform/v1/inspections/{item['id']}/reference-image"
        public["result_json_url"] = f"/api/platform/v1/inspections/{item['id']}/result"
        feedback = self.database.feedback_for_inspection(item["id"])
        public["feedback_receipt"] = (
            {
                "id": feedback["id"],
                "review_status": feedback["review_status"],
                "issue_tags": feedback["issue_tags"],
                "notes": feedback["notes"],
            }
            if feedback
            else None
        )
        return public

    def inspection_reference_image(self, principal: dict[str, Any], inspection_id: str) -> tuple[bytes, str]:
        inspection = self._authorize_inspection(principal, inspection_id)
        if inspection.get("prediction", {}).get("measurement_mode") != "reference_registration":
            raise PlatformError(404, "REFERENCE_IMAGE_NOT_FOUND", "该检测没有参考图。")
        path = self.private_root / "reference_assets" / inspection_id / "reference.upload"
        if not path.is_file():
            raise PlatformError(404, "REFERENCE_IMAGE_NOT_FOUND", "参考图文件不存在。")
        try:
            with Image.open(path) as image:
                content_type = Image.MIME.get(image.format or "", "application/octet-stream")
        except (UnidentifiedImageError, OSError):
            raise PlatformError(404, "REFERENCE_IMAGE_NOT_FOUND", "参考图文件无法读取。")
        return path.read_bytes(), content_type

    def list_inspections(
        self,
        principal: dict[str, Any],
        limit: int = 100,
        offset: int = 0,
        states: list[str] | tuple[str, ...] | None = None,
    ) -> list[dict[str, Any]]:
        if principal["role"] == "admin":
            self.require_owner(principal)
        tenant_id = principal["tenant"]["id"] if principal["role"] == "enterprise" else None
        return [
            self.inspection(principal, item["id"])
            for item in self.database.list_inspections(
                tenant_id, limit=limit, offset=offset, states=states
            )
        ]

    def inspection_page(
        self,
        principal: dict[str, Any],
        *,
        limit: int = 50,
        offset: int = 0,
        status: str = "",
    ) -> dict[str, Any]:
        status_states = {
            "": None,
            "open": ("completed", "detection_failed"),
            "completed": ("completed",),
            "detection_failed": ("detection_failed",),
            "confirmed": ("confirmed",),
            "feedback": (
                "feedback_pending",
                "feedback_approved",
                "feedback_needs_annotation",
                "feedback_discarded",
                "feedback_rejected",
            ),
        }
        if status not in status_states:
            raise PlatformError(422, "INVALID_INSPECTION_STATUS", "检测记录筛选状态无效。")
        tenant_id = principal["tenant"]["id"] if principal["role"] == "enterprise" else None
        if principal["role"] == "admin":
            self.require_owner(principal)
        states = status_states[status]
        items = self.list_inspections(
            principal, limit=limit, offset=offset, states=states
        )
        total = self.database.count_inspections(tenant_id, states=states)
        return {
            "inspections": items,
            "pagination": {
                "limit": int(limit),
                "offset": int(offset),
                "count": len(items),
                "total": total,
                "has_more": int(offset) + len(items) < total,
            },
        }

    def admin_inspections(self, principal: dict[str, Any], limit: int = 500) -> dict[str, Any]:
        self.require_owner(principal)
        items = self.list_inspections(principal, limit=limit)
        tenants = [
            {"id": item["id"], "name": item["name"]}
            for item in self.database.list_tenants()
        ]
        return {
            "inspections": items,
            "tenants": tenants,
            "summary": self.database.summary(),
        }

    def inspection_image(
        self, principal: dict[str, Any], inspection_id: str, variant: str
    ) -> tuple[bytes, str, str]:
        inspection = self._authorize_inspection(principal, inspection_id)
        try:
            if variant == "rectified":
                corners = inspection["prediction"].get("outer_corners")
                if not corners:
                    raise PlatformError(409, "RECTIFICATION_UNAVAILABLE", "外框检测失败，无法生成校正图。")
                data = self.store.rectified_bytes(
                    inspection["project_id"], inspection["sample_id"], corners, 630, 880
                )
                return data, "image/png", f"{inspection_id}_rectified.png"
            return self.store.image_bytes(
                inspection["project_id"], inspection["sample_id"], variant
            )
        except StudioError as exc:
            raise PlatformError(exc.status, exc.code, exc.message, exc.details) from exc

    def inspection_display_image(
        self, principal: dict[str, Any], inspection_id: str, variant: str
    ) -> tuple[bytes, str, str]:
        """Return a compact browser-only image without changing training assets.

        Full-resolution normalized/original PNG files remain untouched for
        inference, export and precision annotation.  The enterprise result
        browser receives a cached WebP derivative so remote review is not
        bottlenecked by multi-megabyte PNG transfers.
        """
        if variant not in {"preview", "rectified"}:
            raise PlatformError(422, "INVALID_DISPLAY_VARIANT", "网页预览图片类型无效。")
        self._authorize_inspection(principal, inspection_id)
        cache_path = self.display_cache_root / (
            f"{inspection_id}_{variant}_{DISPLAY_IMAGE_VERSION}_"
            f"{DISPLAY_PREVIEW_MAX_DIMENSION}_q{DISPLAY_WEBP_QUALITY}.webp"
        )
        with self._display_cache_lock:
            if cache_path.is_file() and cache_path.stat().st_size > 0:
                return cache_path.read_bytes(), "image/webp", cache_path.name

        source, source_type, source_name = self.inspection_image(
            principal, inspection_id, variant
        )
        try:
            with Image.open(io.BytesIO(source)) as opened:
                image = opened.convert("RGB")
                if variant == "preview":
                    image.thumbnail(
                        (
                            DISPLAY_PREVIEW_MAX_DIMENSION,
                            DISPLAY_PREVIEW_MAX_DIMENSION,
                        ),
                        Image.Resampling.LANCZOS,
                    )
                output = io.BytesIO()
                image.save(
                    output,
                    "WEBP",
                    quality=DISPLAY_WEBP_QUALITY,
                    method=4,
                )
                encoded = output.getvalue()
        except (OSError, ValueError, UnidentifiedImageError):
            # A browser can still display the original payload.  Falling back
            # is safer than hiding a result when an unusual legacy image is
            # encountered.
            return source, source_type, source_name

        with self._display_cache_lock:
            if not cache_path.is_file():
                temporary = cache_path.with_suffix(
                    f".{threading.get_ident()}.tmp"
                )
                temporary.write_bytes(encoded)
                temporary.replace(cache_path)
            else:
                encoded = cache_path.read_bytes()
        return encoded, "image/webp", cache_path.name

    def _save_accepted_label(
        self,
        inspection: dict[str, Any],
        *,
        labeler: str,
        reviewer: str,
        corrected_inner: dict[str, float] | None,
        corrected_outer: list[list[float]] | None,
        approve_training: bool,
        issue_tags: list[str],
        notes: str,
    ) -> tuple[dict[str, Any], bool]:
        prediction = inspection["prediction"]
        predicted_outer = prediction.get("outer_corners")
        predicted_centers = prediction.get("inner_line_centers_px")
        if not predicted_outer or not isinstance(predicted_centers, dict):
            raise PlatformError(409, "PREDICTION_INCOMPLETE", "模型结果不完整，必须转入人工高级标注。")
        outer = corrected_outer or _validate_outer(predicted_outer, prediction.get("source_size"))
        assert outer is not None
        centers = corrected_inner or _validate_centers(predicted_centers)
        assert centers is not None
        inner_changed = any(abs(float(centers[key]) - float(predicted_centers[key])) > 0.05 for key in centers)
        outer_changed = any(
            abs(float(outer[index][axis]) - float(predicted_outer[index][axis])) > 0.05
            for index in range(4)
            for axis in range(2)
        )
        changed = inner_changed or outer_changed
        try:
            current = self.store.get_label(inspection["project_id"], inspection["sample_id"])
            label = copy.deepcopy(current)
            label["annotation_status"] = "accepted"
            label["geometry"].update(
                {
                    "outer_state": "applicable",
                    "outer_corners": copy.deepcopy(outer),
                    "inner_state": "applicable",
                    "inner_lines_rectified": _lines_from_centers(centers),
                }
            )
            label["assessment"] = {
                "planarity": "planar",
                "reason_codes": [],
                "shadow_sides": [],
                "notes": notes,
            }
            label["classification"].update({"face": "front", "layout_id": "gx_current", "orientation_degrees_cw": 0})
            metadata = copy.deepcopy(label.get("custom_metadata") or {})
            metadata["platform"] = {
                "inspection_id": inspection["id"],
                "tenant_id": inspection["tenant_id"],
                "model_version": inspection.get("model_version"),
                "confirmed_by_enterprise": not approve_training,
            }
            if approve_training:
                metadata["ml_feedback"] = {
                    "approved_for_training": True,
                    "issue_tags": issue_tags,
                    "source": "cardscope_platform_admin_review",
                }
            label["custom_metadata"] = metadata
            label["annotation"].update(
                {
                    "labeler": labeler,
                    "reviewer": reviewer,
                    "outer_source": "manual" if outer_changed else "prelabel_accepted",
                    "inner_source": "manual" if inner_changed else "prelabel_accepted",
                    "preannotation_disposition": "accepted_modified" if changed else "accepted_unchanged",
                    "prelabel_sha256": prediction.get("prelabel_sha256"),
                }
            )
            return self.store.save_label(
                inspection["project_id"],
                inspection["sample_id"],
                label,
                int(current["annotation"]["revision"]),
            )
        except StudioError as exc:
            raise PlatformError(exc.status, exc.code, exc.message, exc.details) from exc

    def confirm(self, principal: dict[str, Any], inspection_id: str) -> dict[str, Any]:
        self.require_role(principal, "enterprise")
        inspection = self._authorize_inspection(principal, inspection_id)
        self._save_accepted_label(
            inspection,
            labeler=principal["tenant"]["name"],
            reviewer="",
            corrected_inner=None,
            corrected_outer=None,
            approve_training=False,
            issue_tags=[],
            notes="企业端确认模型检测结果。",
        )
        self.database.mark_confirmed(inspection_id)
        return self.inspection(principal, inspection_id)

    def submit_feedback(
        self, principal: dict[str, Any], inspection_id: str, payload: Any
    ) -> dict[str, Any]:
        self.require_role(principal, "enterprise")
        inspection = self._authorize_inspection(principal, inspection_id)
        if not isinstance(payload, dict):
            raise PlatformError(422, "INVALID_FEEDBACK", "反馈内容格式不正确。")
        issue_tags = payload.get("issue_tags")
        if not isinstance(issue_tags, list) or not issue_tags:
            raise PlatformError(422, "ISSUE_TAG_REQUIRED", "请至少选择一个问题类型。")
        normalized_tags: list[str] = []
        for value in issue_tags:
            tag = str(value).strip()
            if tag not in ISSUE_TAGS:
                raise PlatformError(422, "INVALID_ISSUE_TAG", "反馈中包含未知的问题类型。")
            if tag not in normalized_tags:
                normalized_tags.append(tag)
        notes = str(payload.get("notes") or "").strip()
        if len(notes) > 2000:
            raise PlatformError(422, "NOTES_TOO_LONG", "反馈说明不能超过 2000 个字符。")
        corrected = _validate_centers(payload.get("corrected_inner"))
        corrected_outer = _validate_outer(
            payload.get("corrected_outer"), inspection.get("prediction", {}).get("source_size")
        )
        feedback = self.database.submit_feedback(
            inspection_id, normalized_tags, notes, corrected, corrected_outer
        )
        return {
            "feedback": feedback,
            "inspection": self.inspection(principal, inspection_id),
            "message": f"反馈已入库（{feedback['id']}），等待内部人工复核。",
        }

    def admin_feedback(self, principal: dict[str, Any], status: str | None = None) -> dict[str, Any]:
        self.require_annotation_access(principal)
        if self.admin_access_level(principal) == "annotator":
            status = "pending"
        if status and status not in {"pending", "approved", "needs_annotation", "discarded", "rejected"}:
            raise PlatformError(422, "INVALID_REVIEW_STATUS", "反馈状态筛选值无效。")
        items = self.database.list_feedback(status=status)
        for item in items:
            item["images"] = {
                "preview": f"/api/platform/v1/inspections/{item['inspection_id']}/image?variant=preview",
                "normalized": f"/api/platform/v1/inspections/{item['inspection_id']}/image?variant=normalized",
                "rectified": f"/api/platform/v1/inspections/{item['inspection_id']}/image?variant=rectified",
            }
        return {"feedback": items, "summary": self.database.summary()}

    def review_feedback(
        self, principal: dict[str, Any], feedback_id: str, payload: Any
    ) -> dict[str, Any]:
        self.require_annotation_access(principal)
        if not isinstance(payload, dict):
            raise PlatformError(422, "INVALID_REVIEW", "审核内容格式不正确。")
        action = str(payload.get("action") or "")
        if self.admin_access_level(principal) == "annotator" and action not in {"approve", "discard"}:
            raise PlatformError(
                403,
                "ANNOTATOR_ACTION_FORBIDDEN",
                "标注管理员只能提交重新标注结果或舍弃待审核样本，不能驳回、退回或永久删除数据。",
            )
        if action not in {"approve", "discard", "needs_annotation", "reject", "reopen"}:
            raise PlatformError(422, "INVALID_REVIEW_ACTION", "请选择有效的审核操作。")
        feedback = self.database.feedback(feedback_id)
        if feedback is None:
            raise PlatformError(404, "FEEDBACK_NOT_FOUND", "反馈记录不存在。")
        notes = str(payload.get("review_notes") or "").strip()[:2000]
        reviewer = str(principal["admin"]["label"])
        if action == "reopen":
            if feedback.get("review_status") != "approved":
                raise PlatformError(409, "FEEDBACK_NOT_APPROVED", "只有已批准的反馈可以退回待审核。")
            exported_id = str(feedback.get("exported_feedback_id") or "")
            try:
                revocation = self.store.revoke_ml_feedback(
                    feedback["project_id"],
                    feedback["sample_id"],
                    exported_id or None,
                    reviewer=reviewer,
                    reason=notes or "管理员发现已批准标注仍需修正，退回待审核。",
                )
            except StudioError as exc:
                raise PlatformError(exc.status, exc.code, exc.message, exc.details) from exc
            self.database.reopen_feedback(
                feedback_id,
                reviewer=reviewer,
                notes=notes or "已批准结果退回待审核，等待重新标注。",
            )
            return {
                "feedback": self.database.feedback(feedback_id),
                "training_feedback_revocation": revocation,
            }
        if feedback.get("review_status") != "pending":
            raise PlatformError(409, "FEEDBACK_ALREADY_REVIEWED", "这条反馈已经处理，请刷新审核列表。")
        if action != "approve":
            status = "needs_annotation" if action == "needs_annotation" else "discarded" if action == "discard" else "rejected"
            if action == "discard" and not notes:
                notes = "标注管理员判断该样本不适合用于训练，已舍弃。"
            # Rejecting or forwarding a sample must never be blocked by invalid geometry.
            # Preserve any previously saved corrections, and only accept the current draft
            # when it independently passes validation.
            corrected = feedback.get("corrected_inner")
            corrected_outer = feedback.get("corrected_outer")
            try:
                if payload.get("corrected_inner") is not None:
                    corrected = _validate_centers(payload.get("corrected_inner"))
            except PlatformError:
                pass
            try:
                if payload.get("corrected_outer") is not None:
                    corrected_outer = _validate_outer(
                        payload.get("corrected_outer"), feedback.get("prediction", {}).get("source_size")
                    )
            except PlatformError:
                pass
            self.database.review_feedback(
                feedback_id,
                status=status,
                reviewer=reviewer,
                notes=notes,
                corrected_inner=corrected,
                corrected_outer=corrected_outer,
            )
            return {"feedback": self.database.feedback(feedback_id)}
        corrected = _validate_centers(payload.get("corrected_inner", feedback.get("corrected_inner")))
        source_size = feedback.get("prediction", {}).get("source_size")
        corrected_outer = _validate_outer(
            payload.get("corrected_outer", feedback.get("corrected_outer")), source_size
        )
        if corrected is None:
            raise PlatformError(422, "CORRECTION_REQUIRED", "批准训练前必须确认四条内框修正线。")
        if corrected_outer is None:
            corrected_outer = _validate_outer(feedback.get("prediction", {}).get("outer_corners"), source_size)
        if corrected_outer is None:
            raise PlatformError(422, "OUTER_CORRECTION_REQUIRED", "批准训练前必须确认四个外框角点。")
        inspection = self.database.inspection(feedback["inspection_id"])
        if inspection is None:
            raise PlatformError(404, "INSPECTION_NOT_FOUND", "关联检测记录不存在。")
        label, _ = self._save_accepted_label(
            inspection,
            labeler=feedback["tenant_name"],
            reviewer=reviewer,
            corrected_inner=corrected,
            corrected_outer=corrected_outer,
            approve_training=True,
            issue_tags=feedback["issue_tags"],
            notes=feedback["notes"],
        )
        try:
            exported = self.store.export_ml_feedback(
                inspection["project_id"],
                inspection["sample_id"],
                int(label["annotation"]["revision"]),
            )
        except StudioError as exc:
            raise PlatformError(exc.status, exc.code, exc.message, exc.details) from exc
        exported_id = str(exported.get("sample_id") or "")
        self.database.review_feedback(
            feedback_id,
            status="approved",
            reviewer=reviewer,
            notes=notes,
            corrected_inner=corrected,
            corrected_outer=corrected_outer,
            exported_feedback_id=exported_id,
        )
        automatic_job = self._maybe_schedule_auto_training()
        return {
            "feedback": self.database.feedback(feedback_id),
            "training_feedback": exported,
            "automatic_training_job": automatic_job,
        }

    def delete_feedback(
        self,
        principal: dict[str, Any],
        feedback_id: str,
        payload: Any,
    ) -> dict[str, Any]:
        self.require_owner(principal)
        self._require_deletion_idle()
        feedback = self.database.feedback(feedback_id)
        if feedback is None:
            raise PlatformError(404, "FEEDBACK_NOT_FOUND", "反馈记录不存在。")
        if not isinstance(payload, dict) or str(payload.get("confirm_feedback_id") or "") != feedback_id:
            raise PlatformError(422, "FEEDBACK_DELETE_CONFIRMATION_REQUIRED", "删除确认信息不正确。")
        deleted = self.database.delete_feedback_inspection(feedback_id)
        cleanup_warning = None
        cleanup: dict[str, Any] = {}
        try:
            cleanup["training_feedback"] = self.store.delete_ml_feedback_for_sample(
                str(deleted["project_id"]),
                str(deleted["sample_id"]),
                str(deleted.get("exported_feedback_id") or "") or None,
            )
            cleanup["sample"] = self.store.delete_samples(
                str(deleted["project_id"]), {"sample_ids": [str(deleted["sample_id"])]}
            )
        except (StudioError, OSError) as exc:
            cleanup_warning = f"审核记录已删除，但残留文件清理失败：{exc}"
        return {
            "deleted": True,
            "feedback_id": feedback_id,
            "inspection_id": deleted["inspection_id"],
            "filename": deleted["filename"],
            "removed_from_training_pool": bool(deleted.get("exported_feedback_id")),
            "cleanup": cleanup,
            "cleanup_warning": cleanup_warning,
            "summary": self.database.summary(),
        }

    def admin_rectified_preview(
        self, principal: dict[str, Any], feedback_id: str, payload: Any
    ) -> bytes:
        self.require_annotation_access(principal)
        if not isinstance(payload, dict):
            raise PlatformError(422, "INVALID_OUTER_CORRECTION", "外框修正内容格式不正确。")
        feedback = self.database.feedback(feedback_id)
        if feedback is None:
            raise PlatformError(404, "FEEDBACK_NOT_FOUND", "反馈记录不存在。")
        outer = _validate_outer(
            payload.get("corrected_outer"), feedback.get("prediction", {}).get("source_size")
        )
        if outer is None:
            raise PlatformError(422, "OUTER_CORRECTION_REQUIRED", "请先确认四个外框角点。")
        try:
            return self.store.rectified_bytes(
                feedback["project_id"], feedback["sample_id"], outer, 630, 880
            )
        except StudioError as exc:
            raise PlatformError(exc.status, exc.code, exc.message, exc.details) from exc

    def export_feedback_bundle(self, principal: dict[str, Any]) -> tuple[bytes, str]:
        self.require_owner(principal)
        items = self.database.list_feedback(limit=1000)
        manifest = {
            "schema_version": "1.0",
            "created_at": utc_now(),
            "item_count": len(items),
            "description": "CardScope 企业问题样本人工复核包",
        }
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
            rows: list[str] = []
            for item in items:
                public = copy.deepcopy(item)
                rows.append(json_text(public))
                try:
                    image, _, _ = self.store.image_bytes(item["project_id"], item["sample_id"], "normalized")
                    archive.writestr(f"images/{item['id']}.png", image)
                except Exception as exc:
                    archive.writestr(f"errors/{item['id']}.txt", f"normalized image unavailable: {exc}")
                archive.writestr(
                    f"predictions/{item['id']}.json",
                    json.dumps(item.get("prediction") or {}, ensure_ascii=False, indent=2),
                )
            archive.writestr("feedback.jsonl", "\n".join(rows) + ("\n" if rows else ""))
        return output.getvalue(), f"CardScope_feedback_{utc_now()[:10]}.zip"

    @staticmethod
    def _portable_training_yaml(dataset_root: Path) -> None:
        path = dataset_root / "data.yaml"
        if not path.is_file():
            return
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text(
            "\n".join("path: ." if line.startswith("path:") else line for line in lines) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _append_historical_inner(training_root: Path, limit: int) -> dict[str, Any]:
        history_root = Path(__file__).resolve().parents[1] / "training_history" / "inner_seg"
        manifest_path = history_root / "manifest.csv"
        if limit <= 0:
            return {"requested": 0, "included": 0, "available": manifest_path.is_file()}
        if not manifest_path.is_file():
            raise PlatformError(409, "HISTORY_NOT_CONFIGURED", "历史训练数据尚未配置，无法加入导出包。")
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if row.get("split") == "train"]
        selected = rows[: max(0, min(int(limit), 500))]
        image_dir = training_root / "inner_seg" / "images" / "train"
        label_dir = training_root / "inner_seg" / "labels" / "train"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        refiner_path = training_root / "inner_refiner_manifest.csv"
        refiner_fields = ["id", "image", "width", "height", "left", "right", "top", "bottom", "split", "source"]
        existing_rows: list[dict[str, Any]] = []
        if refiner_path.is_file():
            with refiner_path.open("r", encoding="utf-8-sig", newline="") as handle:
                existing_rows = list(csv.DictReader(handle))
        included = 0
        for row in selected:
            source_image = history_root / str(row.get("image") or "")
            source_label = history_root / str(row.get("label") or "")
            if not source_image.is_file() or not source_label.is_file():
                continue
            sample_id = f"history_{row['id']}"
            target_image = image_dir / f"{sample_id}{source_image.suffix.lower()}"
            target_label = label_dir / f"{sample_id}.txt"
            shutil.copy2(source_image, target_image)
            shutil.copy2(source_label, target_label)
            existing_rows.append(
                {
                    "id": sample_id,
                    "image": target_image.relative_to(training_root).as_posix(),
                    "width": row.get("width") or "",
                    "height": row.get("height") or "",
                    "left": row.get("left") or "",
                    "right": row.get("right") or "",
                    "top": row.get("top") or "",
                    "bottom": row.get("bottom") or "",
                    "split": "train",
                    "source": "historical_human_label",
                }
            )
            included += 1
        with refiner_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=refiner_fields)
            writer.writeheader()
            writer.writerows(existing_rows)
        return {"requested": int(limit), "included": included, "available": True}

    @staticmethod
    def _append_historical_outer(training_root: Path, limit: int) -> dict[str, Any]:
        history_root = Path(__file__).resolve().parents[1] / "training_history" / "outer_seg"
        manifest_path = history_root / "manifest.csv"
        if limit <= 0:
            return {"requested": 0, "included": 0, "available": manifest_path.is_file()}
        if not manifest_path.is_file():
            raise PlatformError(409, "HISTORY_NOT_CONFIGURED", "历史外框训练数据尚未配置，无法加入导出包。")
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if row.get("split") == "train"]
        selected = rows[: max(0, min(int(limit), 500))]
        seg_image_dir = training_root / "outer_seg" / "images" / "train"
        seg_label_dir = training_root / "outer_seg" / "labels" / "train"
        pose_image_dir = training_root / "outer_pose" / "images" / "train"
        pose_label_dir = training_root / "outer_pose" / "labels" / "train"
        for path in (seg_image_dir, seg_label_dir, pose_image_dir, pose_label_dir):
            path.mkdir(parents=True, exist_ok=True)
        included = 0
        for row in selected:
            source_image = history_root / str(row.get("image") or "")
            source_label = history_root / str(row.get("label") or "")
            if not source_image.is_file() or not source_label.is_file():
                continue
            label_lines = [
                line.strip()
                for line in source_label.read_text(encoding="utf-8-sig").splitlines()
                if line.strip()
            ]
            values = label_lines[0].split() if len(label_lines) == 1 else []
            if len(values) != 9 or values[0] != "0":
                continue
            try:
                coordinates = [float(value) for value in values[1:]]
            except ValueError:
                continue
            if not all(0.0 <= value <= 1.0 for value in coordinates):
                continue
            sample_id = f"history_{row['id']}"
            image_name = f"{sample_id}{source_image.suffix.lower()}"
            shutil.copy2(source_image, seg_image_dir / image_name)
            shutil.copy2(source_image, pose_image_dir / image_name)
            shutil.copy2(source_label, seg_label_dir / f"{sample_id}.txt")
            xs = coordinates[0::2]
            ys = coordinates[1::2]
            bbox = [
                (min(xs) + max(xs)) / 2.0,
                (min(ys) + max(ys)) / 2.0,
                max(xs) - min(xs),
                max(ys) - min(ys),
            ]
            pose_values: list[float] = [0.0, *bbox]
            for x_value, y_value in zip(xs, ys):
                pose_values.extend([x_value, y_value, 2.0])
            (pose_label_dir / f"{sample_id}.txt").write_text(
                " ".join(f"{value:.8f}" for value in pose_values) + "\n",
                encoding="utf-8",
            )
            included += 1
        return {"requested": int(limit), "included": included, "available": True}

    def export_training_bundle(
        self, principal: dict[str, Any], *, history_limit: int = 0
    ) -> tuple[bytes, str]:
        self.require_owner(principal)
        approved = [
            item
            for item in self.database.list_feedback(status="approved", limit=1000)
            if item.get("exported_feedback_id")
        ]
        if not approved:
            raise PlatformError(409, "NO_APPROVED_FEEDBACK", "还没有审核通过并完成标注的反馈，暂时不能导出训练数据。")
        with tempfile.TemporaryDirectory(prefix="cardscope_training_") as directory:
            temporary = Path(directory)
            feedback_root = temporary / "feedback"
            (feedback_root / "annotations").mkdir(parents=True)
            copied = 0
            for item in approved:
                feedback_id = str(item["exported_feedback_id"])
                source_root = Path(self.store.root) / "ml_feedback" / item["project_id"]
                source_annotation = source_root / "annotations" / f"{feedback_id}.json"
                if not source_annotation.is_file():
                    continue
                annotation = json.loads(source_annotation.read_text(encoding="utf-8"))
                source_image = source_root / str(annotation.get("image", {}).get("path") or "")
                source_rectified = source_root / str(annotation.get("rectification", {}).get("image_path") or "")
                if not source_image.is_file() or not source_rectified.is_file():
                    continue
                image_target = feedback_root / "original_images" / f"{feedback_id}{source_image.suffix.lower()}"
                rectified_target = feedback_root / "rectified_images" / f"{feedback_id}{source_rectified.suffix.lower()}"
                image_target.parent.mkdir(parents=True, exist_ok=True)
                rectified_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_image, image_target)
                shutil.copy2(source_rectified, rectified_target)
                annotation["image"]["path"] = image_target.relative_to(feedback_root).as_posix()
                annotation["rectification"]["image_path"] = rectified_target.relative_to(feedback_root).as_posix()
                (feedback_root / "annotations" / f"{feedback_id}.json").write_text(
                    json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                copied += 1
            if not copied:
                raise PlatformError(409, "TRAINING_SOURCE_MISSING", "审核记录存在，但训练反馈文件缺失，请联系管理员检查存储目录。")
            ml_root = Path(__file__).resolve().parents[1] / "ml_backend"
            if str(ml_root) not in sys.path:
                sys.path.insert(0, str(ml_root))
            try:
                from feedback_dataset import convert_feedback_to_training

                training_root = temporary / "training_data"
                conversion = convert_feedback_to_training(feedback_root, training_root, split="train")
            except (OSError, TypeError, ValueError) as exc:
                raise PlatformError(500, "TRAINING_EXPORT_FAILED", f"训练数据转换失败：{exc}") from exc
            inner_history = self._append_historical_inner(training_root, history_limit)
            outer_history = self._append_historical_outer(training_root, history_limit)
            for dataset in ("outer_pose", "outer_seg", "inner_seg"):
                self._portable_training_yaml(training_root / dataset)
            manifest = {
                "schema_version": "1.0",
                "created_at": utc_now(),
                "approved_feedback_records": len(approved),
                "feedback_packages_copied": copied,
                "converted": conversion.get("converted", {}),
                "historical_inner": inner_history,
                "historical_outer": outer_history,
                "history_policy": "仅加入经过校验的历史 train 集；历史 val/test 永不混入训练导出。",
            }
            (training_root / "export_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (training_root / "使用说明.txt").write_text(
                "CardScope 模型训练数据\n\nouter_pose：外框四角点训练集\nouter_seg：外框分割训练集\ninner_seg：内框分割训练集\ninner_refiner_manifest.csv：内框精修训练清单\nexport_manifest.json：本次导出数量与历史数据来源说明\n",
                encoding="utf-8",
            )
            output = io.BytesIO()
            with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
                for path in sorted(training_root.rglob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(training_root).as_posix())
            return output.getvalue(), f"CardScope_training_data_{utc_now()[:10]}.zip"


__all__ = ["ISSUE_TAGS", "PlatformError", "PlatformService"]

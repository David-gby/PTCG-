from __future__ import annotations

import hashlib
import json
import mimetypes
import re
import traceback
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from .service import PlatformError, PlatformService


INSPECTION_ROUTE = re.compile(r"^/api/platform/v1/inspections/(ins_[a-z0-9_]+)$")
RESULT_ROUTE = re.compile(r"^/api/platform/v1/inspections/(ins_[a-z0-9_]+)/result$")
REFERENCE_IMAGE_ROUTE = re.compile(r"^/api/platform/v1/inspections/(ins_[a-z0-9_]+)/reference-image$")
REFERENCE_JOB_ROUTE = re.compile(r"^/api/platform/v1/reference-inspections/(reg_[a-z0-9_]+)$")
REFERENCE_FILE_ROUTE = re.compile(r"^/api/platform/v1/reference-inspections/(reg_[a-z0-9_]+)/(capture|reference)$")
IMAGE_ROUTE = re.compile(r"^/api/platform/v1/inspections/(ins_[a-z0-9_]+)/image$")
CONFIRM_ROUTE = re.compile(r"^/api/platform/v1/inspections/(ins_[a-z0-9_]+)/confirm$")
FEEDBACK_ROUTE = re.compile(r"^/api/platform/v1/inspections/(ins_[a-z0-9_]+)/feedback$")
BATCH_ROUTE = re.compile(r"^/api/platform/v1/batches/(bat_[a-z0-9_]+)$")
BATCH_ITEM_ROUTE = re.compile(
    r"^/api/platform/v1/batches/(bat_[a-z0-9_]+)/items/(bti_[a-z0-9_]+)$"
)
BATCH_ACTION_ROUTE = re.compile(
    r"^/api/platform/v1/batches/(bat_[a-z0-9_]+)/action$"
)
ADMIN_REVIEW_ROUTE = re.compile(r"^/api/platform/v1/admin/feedback/(fbk_[a-z0-9_]+)/review$")
ADMIN_RECTIFY_ROUTE = re.compile(r"^/api/platform/v1/admin/feedback/(fbk_[a-z0-9_]+)/rectify-preview$")
ADMIN_FEEDBACK_ROUTE = re.compile(r"^/api/platform/v1/admin/feedback/(fbk_[a-z0-9_]+)$")
ADMIN_TENANT_ACTION_ROUTE = re.compile(
    r"^/api/platform/v1/admin/tenants/(ten_[a-z0-9_]+)/action$"
)
ADMIN_TENANT_ROUTE = re.compile(r"^/api/platform/v1/admin/tenants/(ten_[a-z0-9_]+)$")
ADMIN_USER_ACTION_ROUTE = re.compile(
    r"^/api/platform/v1/admin/users/(admusr_[a-z0-9_]+)/action$"
)
ADMIN_USER_ROUTE = re.compile(r"^/api/platform/v1/admin/users/(admusr_[a-z0-9_]+)$")
SESSION_COOKIE = "cardscope_session"


class PlatformHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        service: PlatformService,
        web_root: Path,
        max_upload_bytes: int,
        max_json_bytes: int,
        instance_id: str = "",
        release_version: str = "",
    ) -> None:
        super().__init__(address, PlatformHandler)
        self.service = service
        self.web_root = web_root.resolve()
        self.max_upload_bytes = max_upload_bytes
        self.max_json_bytes = max_json_bytes
        self.instance_id = instance_id
        self.release_version = release_version


class PlatformHandler(BaseHTTPRequestHandler):
    server: PlatformHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        path = urlsplit(self.path).path
        print(f"{self.client_address[0]} - {self.command} {path} - {fmt % args}")

    def _security_headers(self, *, cache: str = "no-store") -> None:
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' blob: data:; style-src 'self'; "
            "script-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; "
            "frame-ancestors 'none'; form-action 'self'",
        )

    def _send_bytes(
        self,
        status: int,
        data: bytes,
        content_type: str,
        *,
        filename: str | None = None,
        cache: str = "no-store",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self._security_headers(cache=cache)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if filename:
            safe = filename.replace('"', "_").replace("\r", "_").replace("\n", "_")
            self.send_header("Content-Disposition", f'attachment; filename="{safe}"')
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _send_json(
        self, status: int, value: Any, *, headers: dict[str, str] | None = None
    ) -> None:
        self._send_bytes(
            status,
            json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"),
            "application/json; charset=utf-8",
            headers=headers,
        )

    def _send_cacheable_image(
        self,
        data: bytes,
        content_type: str,
        *,
        cache: str,
    ) -> None:
        etag = f'"{hashlib.sha256(data).hexdigest()}"'
        headers = {"ETag": etag, "Vary": "Cookie"}
        if self.headers.get("If-None-Match", "").strip() == etag:
            self.send_response(304)
            self._security_headers(cache=cache)
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            return
        self._send_bytes(
            200,
            data,
            content_type,
            filename=None,
            cache=cache,
            headers=headers,
        )

    def _send_error(self, error: PlatformError) -> None:
        self._send_json(
            error.status,
            {"error": {"code": error.code, "message": error.message, "details": error.details}},
        )

    def _parsed(self):
        return urlsplit(self.path)

    def _query(self) -> dict[str, list[str]]:
        return parse_qs(self._parsed().query, keep_blank_values=True)

    def _token(self) -> str:
        header = self.headers.get("X-Platform-Token", "").strip()
        query = self._query().get("access", [""])[0].strip()
        return header or query

    def _session_token(self) -> str:
        raw = self.headers.get("Cookie", "")
        if not raw or len(raw) > 4096:
            return ""
        cookie = SimpleCookie()
        try:
            cookie.load(raw)
        except CookieError:
            return ""
        morsel = cookie.get(SESSION_COOKIE)
        return morsel.value if morsel else ""

    def _secure_cookie(self) -> bool:
        forwarded = self.headers.get("X-Forwarded-Proto", "").split(",", 1)[0].strip().lower()
        return forwarded == "https"

    def _set_session_cookie(self, token: str, maximum_age: int = 12 * 60 * 60) -> str:
        parts = [
            f"{SESSION_COOKIE}={token}",
            "Path=/",
            f"Max-Age={maximum_age}",
            "HttpOnly",
            "SameSite=Lax",
        ]
        if self._secure_cookie():
            parts.append("Secure")
        return "; ".join(parts)

    def _clear_session_cookie(self) -> str:
        parts = [
            f"{SESSION_COOKIE}=",
            "Path=/",
            "Max-Age=0",
            "HttpOnly",
            "SameSite=Lax",
        ]
        if self._secure_cookie():
            parts.append("Secure")
        return "; ".join(parts)

    def _principal(self) -> dict[str, Any]:
        token = self._token()
        if token:
            return self.server.service.authenticate(token)
        return self.server.service.authenticate_session(self._session_token())

    def _validate_origin(self) -> None:
        origin = self.headers.get("Origin")
        if not origin:
            return
        expected_http = f"http://{self.headers.get('Host', '')}"
        expected_https = f"https://{self.headers.get('Host', '')}"
        if origin not in {expected_http, expected_https}:
            raise PlatformError(403, "ORIGIN_FORBIDDEN", "请求来源不受信任。")

    def _read_body(self, maximum: int) -> bytes:
        raw = self.headers.get("Content-Length")
        if raw is None:
            raise PlatformError(411, "CONTENT_LENGTH_REQUIRED", "请求缺少内容长度。")
        try:
            length = int(raw)
        except ValueError as exc:
            raise PlatformError(400, "INVALID_CONTENT_LENGTH", "内容长度格式不正确。") from exc
        if length < 0 or length > maximum:
            raise PlatformError(413, "PAYLOAD_TOO_LARGE", "上传内容超过平台限制。")
        return self.rfile.read(length)

    def _read_json(self) -> Any:
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise PlatformError(415, "JSON_REQUIRED", "此操作需要 JSON 请求。")
        try:
            return json.loads(self._read_body(self.server.max_json_bytes).decode("utf-8"))
        except (UnicodeError, ValueError) as exc:
            raise PlatformError(400, "INVALID_JSON", "JSON 内容无法解析。") from exc

    def _serve_static(self, name: str) -> None:
        mapping = {
            "/": "access.html",
            "/login": "access.html",
            "/admin-login": "admin_login.html",
            "/enterprise": "enterprise.html",
            "/admin": "admin.html",
            "/assets/styles.css": "styles.css",
            "/assets/common.js": "common.js",
            "/assets/login.js": "login.js",
            "/assets/admin-login.js": "admin_login.js",
            "/assets/enterprise.js": "enterprise.js",
            "/assets/reference_result_view.js": "reference_result_view.js",
            "/assets/admin.js": "admin.js",
        }
        target = mapping.get(name)
        if target is None:
            raise PlatformError(404, "NOT_FOUND", "页面不存在。")
        path = (self.server.web_root / target).resolve()
        if path.parent != self.server.web_root or not path.is_file():
            raise PlatformError(404, "NOT_FOUND", "页面资源不存在。")
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in {"application/javascript", "application/json"}:
            content_type += "; charset=utf-8"
        self._send_bytes(200, path.read_bytes(), content_type, cache="no-cache")

    def do_GET(self) -> None:
        try:
            path = self._parsed().path
            if path in {"/", "/login", "/admin-login", "/enterprise", "/admin"} or path.startswith("/assets/"):
                self._serve_static(path)
                return
            if path == "/api/platform/v1/health":
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "app": "CardScope",
                        "api_version": "1.0",
                        "instance_id": self.server.instance_id,
                        "release_version": self.server.release_version,
                    },
                )
                return
            principal = self._principal()
            if path == "/api/platform/v1/session":
                self._send_json(200, {"session": self.server.service.session(principal)})
                return
            if path == "/api/platform/v1/inspections":
                query = self._query()
                try:
                    limit = max(1, min(int(query.get("limit", ["50"])[0]), 100))
                    offset = max(0, int(query.get("offset", ["0"])[0]))
                except ValueError as exc:
                    raise PlatformError(
                        422,
                        "INVALID_INSPECTION_PAGE",
                        "检测记录分页参数格式不正确。",
                    ) from exc
                status = query.get("status", [""])[0]
                self._send_json(
                    200,
                    self.server.service.inspection_page(
                        principal, limit=limit, offset=offset, status=status
                    ),
                )
                return
            if path == "/api/platform/v1/batches":
                raw_limit = self._query().get("limit", ["20"])[0]
                try:
                    limit = max(1, min(int(raw_limit), 100))
                except ValueError as exc:
                    raise PlatformError(
                        422, "INVALID_BATCH_LIMIT", "批量任务数量格式不正确。"
                    ) from exc
                self._send_json(
                    200,
                    {"batches": self.server.service.list_batch_jobs(principal, limit)},
                )
                return
            match = BATCH_ROUTE.fullmatch(path)
            if match:
                include_results = self._query().get("include", [""])[0] == "results"
                self._send_json(
                    200,
                    {
                        "batch": self.server.service.batch_job(
                            principal,
                            match.group(1),
                            include_inspections=include_results,
                        )
                    },
                )
                return
            match = IMAGE_ROUTE.fullmatch(path)
            if match:
                variant = self._query().get("variant", ["preview"])[0]
                if variant not in {"preview", "original", "normalized", "rectified"}:
                    raise PlatformError(422, "INVALID_IMAGE_VARIANT", "图片类型无效。")
                display = self._query().get("display", [""])[0]
                if display not in {"", "webp"}:
                    raise PlatformError(422, "INVALID_DISPLAY_FORMAT", "网页预览格式无效。")
                if display == "webp":
                    data, content_type, filename = (
                        self.server.service.inspection_display_image(
                            principal, match.group(1), variant
                        )
                    )
                    cache = "private, max-age=86400, immutable"
                else:
                    data, content_type, filename = self.server.service.inspection_image(
                        principal, match.group(1), variant
                    )
                    cache = "private, max-age=3600"
                self._send_cacheable_image(data, content_type, cache=cache)
                return
            match_reference_image = REFERENCE_IMAGE_ROUTE.fullmatch(path)
            if match_reference_image:
                data, content_type = self.server.service.inspection_reference_image(principal, match_reference_image.group(1))
                self._send_cacheable_image(data, content_type, cache="private, max-age=3600")
                return
            match_reference_job = REFERENCE_JOB_ROUTE.fullmatch(path)
            if match_reference_job:
                self._send_json(200, {"job": self.server.service.reference_job(principal, match_reference_job.group(1))})
                return
            match = INSPECTION_ROUTE.fullmatch(path)
            match_result = RESULT_ROUTE.fullmatch(path)
            if match_result:
                inspection = self.server.service._authorize_inspection(principal, match_result.group(1))
                from .result_exports import inspection_result_path

                result_path = inspection_result_path(self.server.service.workspace, inspection["id"])
                if not result_path.is_file():
                    raise PlatformError(404, "RESULT_NOT_FOUND", "检测结果文件不存在。")
                self._send_bytes(
                    200,
                    result_path.read_bytes(),
                    "application/json; charset=utf-8",
                    filename=f"{inspection['id']}-result.json",
                )
                return
            if match:
                self._send_json(200, {"inspection": self.server.service.inspection(principal, match.group(1))})
                return
            if path == "/api/platform/v1/admin/feedback":
                status = self._query().get("status", [""])[0] or None
                self._send_json(200, self.server.service.admin_feedback(principal, status))
                return
            if path == "/api/platform/v1/admin/inspections":
                raw_limit = self._query().get("limit", ["500"])[0]
                try:
                    limit = max(1, min(int(raw_limit), 500))
                except ValueError as exc:
                    raise PlatformError(422, "INVALID_INSPECTION_LIMIT", "检测记录数量格式不正确。") from exc
                self._send_json(200, self.server.service.admin_inspections(principal, limit))
                return
            if path == "/api/platform/v1/admin/tenants":
                self._send_json(200, self.server.service.admin_tenants(principal))
                return
            if path == "/api/platform/v1/admin/users":
                self._send_json(200, self.server.service.admin_users(principal))
                return
            if path == "/api/platform/v1/admin/training":
                self._send_json(200, self.server.service.admin_training_status(principal))
                return
            if path == "/api/platform/v1/admin/feedback-export":
                data, filename = self.server.service.export_feedback_bundle(principal)
                self._send_bytes(200, data, "application/zip", filename=filename)
                return
            if path == "/api/platform/v1/admin/training-export":
                raw_limit = self._query().get("history_limit", ["0"])[0]
                try:
                    history_limit = max(0, min(int(raw_limit), 500))
                except ValueError as exc:
                    raise PlatformError(422, "INVALID_HISTORY_LIMIT", "历史样本数量格式不正确。") from exc
                data, filename = self.server.service.export_training_bundle(
                    principal, history_limit=history_limit
                )
                self._send_bytes(200, data, "application/zip", filename=filename)
                return
            raise PlatformError(404, "NOT_FOUND", "接口不存在。")
        except PlatformError as error:
            self._send_error(error)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception:
            traceback.print_exc()
            self._send_error(PlatformError(500, "INTERNAL_ERROR", "平台发生内部错误，请联系管理员。"))

    def do_POST(self) -> None:
        try:
            self._validate_origin()
            path = self._parsed().path
            if path == "/api/platform/v1/auth/login":
                result = self.server.service.login_enterprise(self._read_json())
                session_token = str(result.pop("session_token"))
                self._send_json(
                    200,
                    result,
                    headers={"Set-Cookie": self._set_session_cookie(session_token)},
                )
                return
            if path == "/api/platform/v1/auth/logout":
                self.server.service.logout_enterprise(self._session_token())
                self._send_json(
                    200,
                    {"message": "已安全退出。"},
                    headers={"Set-Cookie": self._clear_session_cookie()},
                )
                return
            principal = self._principal()
            if path == "/api/platform/v1/reference-inspections":
                result = self.server.service.create_reference_job(principal, self._read_json())
                self._send_json(201, {"job": result})
                return
            match_reference_file = REFERENCE_FILE_ROUTE.fullmatch(path)
            if match_reference_file:
                content_type = self.headers.get("Content-Type", "application/octet-stream")
                result = self.server.service.upload_reference_file(
                    principal, match_reference_file.group(1), match_reference_file.group(2),
                    self._read_body(self.server.max_upload_bytes), content_type,
                )
                self._send_json(201, result)
                return
            if path == "/api/platform/v1/batches":
                result = self.server.service.create_batch_job(
                    principal, self._read_json()
                )
                self._send_json(201, {"batch": result})
                return
            match = BATCH_ITEM_ROUTE.fullmatch(path)
            if match:
                content_type = self.headers.get("Content-Type", "application/octet-stream")
                payload = self._read_body(self.server.max_upload_bytes)
                result = self.server.service.upload_batch_item(
                    principal,
                    match.group(1),
                    match.group(2),
                    payload,
                    content_type,
                )
                self._send_json(201, result)
                return
            match = BATCH_ACTION_ROUTE.fullmatch(path)
            if match:
                result = self.server.service.batch_job_action(
                    principal, match.group(1), self._read_json()
                )
                self._send_json(200, {"batch": result})
                return
            if path == "/api/platform/v1/inspections":
                content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
                if content_type != "application/octet-stream" and not content_type.startswith("image/"):
                    raise PlatformError(415, "IMAGE_REQUIRED", "请上传 JPG 或 PNG 图片。")
                filename = unquote(self._query().get("filename", [""])[0])
                if not filename:
                    raise PlatformError(422, "FILENAME_REQUIRED", "上传文件缺少文件名。")
                payload = self._read_body(self.server.max_upload_bytes)
                result = self.server.service.create_inspection(principal, filename, payload)
                self._send_json(201, {"inspection": result})
                return
            match = CONFIRM_ROUTE.fullmatch(path)
            if match:
                self._read_json()
                result = self.server.service.confirm(principal, match.group(1))
                self._send_json(200, {"inspection": result})
                return
            match = FEEDBACK_ROUTE.fullmatch(path)
            if match:
                result = self.server.service.submit_feedback(
                    principal, match.group(1), self._read_json()
                )
                self._send_json(201, result)
                return
            match = ADMIN_REVIEW_ROUTE.fullmatch(path)
            if match:
                result = self.server.service.review_feedback(
                    principal, match.group(1), self._read_json()
                )
                self._send_json(200, result)
                return
            match = ADMIN_RECTIFY_ROUTE.fullmatch(path)
            if match:
                data = self.server.service.admin_rectified_preview(
                    principal, match.group(1), self._read_json()
                )
                self._send_bytes(200, data, "image/png", cache="no-store")
                return
            if path == "/api/platform/v1/admin/tenants":
                result = self.server.service.create_tenant(principal, self._read_json())
                self._send_json(201, result)
                return
            if path == "/api/platform/v1/admin/users":
                result = self.server.service.create_admin_user(principal, self._read_json())
                self._send_json(201, result)
                return
            if path == "/api/platform/v1/admin/training/settings":
                result = self.server.service.update_training_settings(principal, self._read_json())
                self._send_json(200, result)
                return
            if path == "/api/platform/v1/admin/training/start":
                result = self.server.service.start_auto_training(principal, self._read_json())
                self._send_json(202, result)
                return
            if path == "/api/platform/v1/admin/training/rollback":
                result = self.server.service.rollback_auto_model(principal, self._read_json())
                self._send_json(200, result)
                return
            match = ADMIN_TENANT_ACTION_ROUTE.fullmatch(path)
            if match:
                result = self.server.service.tenant_action(
                    principal, match.group(1), self._read_json()
                )
                self._send_json(200, result)
                return
            match = ADMIN_USER_ACTION_ROUTE.fullmatch(path)
            if match:
                result = self.server.service.admin_user_action(
                    principal, match.group(1), self._read_json()
                )
                self._send_json(200, result)
                return
            raise PlatformError(404, "NOT_FOUND", "接口不存在。")
        except PlatformError as error:
            self._send_error(error)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception:
            traceback.print_exc()
            self._send_error(PlatformError(500, "INTERNAL_ERROR", "平台发生内部错误，请联系管理员。"))

    def do_DELETE(self) -> None:
        try:
            self._validate_origin()
            path = self._parsed().path
            principal = self._principal()
            match = ADMIN_FEEDBACK_ROUTE.fullmatch(path)
            if match:
                result = self.server.service.delete_feedback(
                    principal, match.group(1), self._read_json()
                )
                self._send_json(200, result)
                return
            match = ADMIN_TENANT_ROUTE.fullmatch(path)
            if match:
                result = self.server.service.delete_tenant(
                    principal, match.group(1), self._read_json()
                )
                self._send_json(200, result)
                return
            match = ADMIN_USER_ROUTE.fullmatch(path)
            if match:
                result = self.server.service.delete_admin_user(
                    principal, match.group(1), self._read_json()
                )
                self._send_json(200, result)
                return
            raise PlatformError(404, "NOT_FOUND", "接口不存在。")
        except PlatformError as error:
            self._send_error(error)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception:
            traceback.print_exc()
            self._send_error(PlatformError(500, "INTERNAL_ERROR", "平台发生内部错误，请联系管理员。"))

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_OPTIONS(self) -> None:
        self._send_error(PlatformError(405, "METHOD_NOT_ALLOWED", "不支持跨站请求。"))


def build_server(
    service: PlatformService,
    host: str,
    port: int,
    web_root: Path,
    max_upload_bytes: int,
    max_json_bytes: int,
    instance_id: str = "",
    release_version: str = "",
) -> PlatformHTTPServer:
    return PlatformHTTPServer(
        (host, port),
        service,
        web_root,
        max_upload_bytes,
        max_json_bytes,
        instance_id,
        release_version,
    )


__all__ = ["build_server"]

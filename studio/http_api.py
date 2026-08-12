from __future__ import annotations

import hmac
import json
import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

from .config import OPTIONAL_FORMATS, RESOURCE_ROOT, SUPPORTED_BASE_FORMATS
from .errors import StudioError
from .exports import build_export
from .security import json_bytes, strict_json_loads
from .store import StudioStore


STATIC_ROOT = RESOURCE_ROOT / "static"
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/static/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/static/app.js": ("app.js", "text/javascript; charset=utf-8"),
}

PROJECT_ROUTE = re.compile(r"^/api/v1/projects/([a-z][a-z0-9_-]{2,95})$")
IMPORTS_ROUTE = re.compile(r"^/api/v1/projects/([a-z][a-z0-9_-]{2,95})/imports$")
IMPORT_ROUTE = re.compile(r"^/api/v1/projects/([a-z][a-z0-9_-]{2,95})/imports/([a-z][a-z0-9_-]{2,95})$")
IMPORT_FILE_ROUTE = re.compile(r"^/api/v1/projects/([a-z][a-z0-9_-]{2,95})/imports/([a-z][a-z0-9_-]{2,95})/files$")
SAMPLES_ROUTE = re.compile(r"^/api/v1/projects/([a-z][a-z0-9_-]{2,95})/samples$")
IMAGE_ROUTE = re.compile(r"^/api/v1/projects/([a-z][a-z0-9_-]{2,95})/samples/([a-z][a-z0-9_-]{2,95})/image$")
RECTIFIED_ROUTE = re.compile(r"^/api/v1/projects/([a-z][a-z0-9_-]{2,95})/samples/([a-z][a-z0-9_-]{2,95})/rectified$")
LABEL_ROUTE = re.compile(r"^/api/v1/projects/([a-z][a-z0-9_-]{2,95})/samples/([a-z][a-z0-9_-]{2,95})/label$")
PRELABEL_ROUTE = re.compile(r"^/api/v1/projects/([a-z][a-z0-9_-]{2,95})/samples/([a-z][a-z0-9_-]{2,95})/prelabel$")
ML_FEEDBACK_ROUTE = re.compile(r"^/api/v1/projects/([a-z][a-z0-9_-]{2,95})/samples/([a-z][a-z0-9_-]{2,95})/ml-feedback$")
TRAINING_JOBS_ROUTE = re.compile(r"^/api/v1/projects/([a-z][a-z0-9_-]{2,95})/training-jobs$")
TRAINING_JOB_ROUTE = re.compile(r"^/api/v1/projects/([a-z][a-z0-9_-]{2,95})/training-jobs/([a-z][a-z0-9_-]{2,95})$")
EXPORT_ROUTE = re.compile(r"^/api/v1/projects/([a-z][a-z0-9_-]{2,95})/export$")
CLIENT_DISCONNECT_ERRORS = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)


class StudioHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, address: tuple[str, int], store: StudioStore, token: str) -> None:
        self.store = store
        self.token = token
        super().__init__(address, StudioHandler)

    @property
    def allowed_hosts(self) -> set[str]:
        port = int(self.server_address[1])
        return {f"127.0.0.1:{port}", f"localhost:{port}"}

    @property
    def allowed_origins(self) -> set[str]:
        return {f"http://{host}" for host in self.allowed_hosts}


class StudioHandler(BaseHTTPRequestHandler):
    server: StudioHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        message = format % args
        print(f"[{self.log_date_time_string()}] {self.client_address[0]} {message}", flush=True)

    def _send_security_headers(self, *, cache_control: str = "no-store") -> None:
        self.send_header("Cache-Control", cache_control)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        if self.close_connection:
            self.send_header("Connection", "close")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' blob: data:; connect-src 'self'; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'",
        )

    def _send_bytes(
        self,
        status: int,
        body: bytes,
        content_type: str,
        *,
        disposition: str | None = None,
        cache_control: str = "no-store",
    ) -> None:
        try:
            self.send_response(status)
            self._send_security_headers(cache_control=cache_control)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            if disposition:
                self.send_header("Content-Disposition", disposition)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
        except CLIENT_DISCONNECT_ERRORS:
            self.close_connection = True

    def _send_json(self, status: int, value: Any) -> None:
        self._send_bytes(status, json_bytes(value, pretty=False), "application/json; charset=utf-8")

    def _send_error(self, error: StudioError) -> None:
        self._send_json(
            error.status,
            {
                "error": {
                    "code": error.code,
                    "message": error.message,
                    "details": error.details,
                }
            },
        )

    def _parsed(self):
        try:
            return urlparse(self.path)
        except ValueError as exc:
            raise StudioError(400, "INVALID_URL", "Malformed request URL.") from exc

    def _validate_host(self) -> None:
        hosts = self.headers.get_all("Host") or []
        if len(hosts) != 1 or hosts[0].lower() not in self.server.allowed_hosts:
            raise StudioError(421, "INVALID_HOST", "Host header is not an allowed local endpoint.")
        if self.client_address[0] not in {"127.0.0.1", "::1"}:
            raise StudioError(403, "LOCAL_ONLY", "Only local clients are allowed.")

    def _validate_framing(self) -> None:
        if self.headers.get("Transfer-Encoding") is not None:
            self.close_connection = True
            raise StudioError(400, "TRANSFER_ENCODING_FORBIDDEN", "Chunked request bodies are not accepted.")
        lengths = self.headers.get_all("Content-Length") or []
        if len(lengths) > 1:
            self.close_connection = True
            raise StudioError(400, "AMBIGUOUS_CONTENT_LENGTH", "Multiple Content-Length headers are forbidden.")
        if lengths:
            try:
                length = int(lengths[0])
            except ValueError as exc:
                raise StudioError(400, "INVALID_CONTENT_LENGTH", "Content-Length is invalid.") from exc
            if length < 0:
                raise StudioError(400, "INVALID_CONTENT_LENGTH", "Content-Length is invalid.")

    def _query(self) -> dict[str, list[str]]:
        try:
            return parse_qs(self._parsed().query, keep_blank_values=True, strict_parsing=False)
        except ValueError as exc:
            raise StudioError(400, "INVALID_QUERY", "Malformed query string.") from exc

    def _require_token(self) -> None:
        query_tokens = self._query().get("token", [])
        header_tokens = self.headers.get_all("X-Studio-Token") or []
        candidates = header_tokens if header_tokens else query_tokens
        if len(candidates) != 1 or not hmac.compare_digest(candidates[0], self.server.token):
            raise StudioError(401, "INVALID_TOKEN", "The local session token is missing or invalid.")

    def _require_origin(self) -> None:
        origins = self.headers.get_all("Origin") or []
        if len(origins) != 1 or origins[0].lower() not in self.server.allowed_origins:
            raise StudioError(403, "INVALID_ORIGIN", "Mutating requests require the exact local Origin.")

    def _require_content_type(self, allowed: set[str], *, image_wildcard: bool = False) -> None:
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type in allowed:
            return
        if image_wildcard and content_type.startswith("image/"):
            return
        raise StudioError(415, "UNSUPPORTED_CONTENT_TYPE", "Request Content-Type is not supported.")

    def _content_length(self) -> int:
        raw = self.headers.get("Content-Length")
        if raw is None:
            raise StudioError(411, "CONTENT_LENGTH_REQUIRED", "Content-Length is required.")
        return int(raw)

    def _read_body(self, maximum: int) -> bytes:
        length = self._content_length()
        if length > maximum:
            self.close_connection = True
            raise StudioError(413, "REQUEST_TOO_LARGE", f"Request exceeds the {maximum} byte limit.")
        body = self.rfile.read(length)
        if len(body) != length:
            self.close_connection = True
            raise StudioError(400, "INCOMPLETE_REQUEST_BODY", "Request body ended early.")
        return body

    def _read_json(self) -> Any:
        self._require_content_type({"application/json"})
        raw = self._read_body(self.server.store.config.max_json_bytes)
        try:
            return strict_json_loads(raw.decode("utf-8"))
        except (UnicodeError, ValueError) as exc:
            raise StudioError(400, "INVALID_JSON", f"Strict JSON parsing failed: {exc}") from exc

    def _bootstrap(self) -> dict[str, Any]:
        config = self.server.store.config
        return {
            "app": "PTCG Annotation Studio ML",
            "api_version": "1.0",
            "limits": {
                "max_upload_bytes": config.max_upload_bytes,
                "max_pixels": config.max_pixels,
                "max_dimension": config.max_dimension,
                "preview_max_dimension": config.preview_max_dimension,
                "max_project_assets": config.max_project_assets,
            },
            "formats": SUPPORTED_BASE_FORMATS,
            "optional_formats": OPTIONAL_FORMATS,
            "optional_codecs": self.server.store.codec_status(),
            "projects": self.server.store.list_projects(),
        }

    def _serve_static(self, path: str) -> None:
        if path not in STATIC_FILES:
            raise StudioError(404, "NOT_FOUND", "Resource does not exist.")
        if path in {"/", "/index.html"}:
            self._require_token()
        filename, content_type = STATIC_FILES[path]
        file_path = STATIC_ROOT / filename
        if not file_path.is_file():
            raise StudioError(503, "UI_NOT_INSTALLED", f"Missing static resource: {filename}")
        self._send_bytes(200, file_path.read_bytes(), content_type)

    @staticmethod
    def _download_disposition(filename: str) -> str:
        ascii_name = "".join(character if ord(character) < 128 and character not in '"\\' else "_" for character in filename)
        return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename)}"

    def _stream_export(
        self,
        project_id: str,
        mode: str,
        sample_ids: list[str] | None = None,
    ) -> None:
        artifact = build_export(self.server.store, project_id, mode, sample_ids)
        try:
            self.send_response(200)
            self._send_security_headers()
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Length", str(artifact.size))
            self.send_header("Content-Disposition", self._download_disposition(artifact.filename))
            self.end_headers()
            with artifact.path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    self.wfile.write(block)
        finally:
            artifact.cleanup()

    def do_GET(self) -> None:
        try:
            self._validate_host()
            self._validate_framing()
            parsed = self._parsed()
            path = unquote(parsed.path)
            if path in STATIC_FILES:
                self._serve_static(path)
                return
            if path == "/api/v1/health":
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "app": "PTCG Annotation Studio ML",
                        "api_version": "1.0",
                        "local_only": True,
                    },
                )
                return
            self._require_token()
            if path == "/api/v1/bootstrap":
                self._send_json(200, self._bootstrap())
                return
            if path == "/api/v1/training-history":
                query = self._query()
                unknown = set(query) - {"token", "targets"}
                values = query.get("targets", [])
                if unknown or len(values) > 1:
                    raise StudioError(400, "INVALID_QUERY", "targets must be supplied at most once.")
                targets = values[0].split(",") if values and values[0] else None
                from .historical_replay import historical_replay_status

                self._send_json(200, {"historical_replay": historical_replay_status(targets, full=False)})
                return
            if path == "/api/v1/projects":
                self._send_json(200, {"projects": self.server.store.list_projects()})
                return
            if match := PROJECT_ROUTE.fullmatch(path):
                self._send_json(200, self.server.store.project_detail(match.group(1)))
                return
            if match := IMPORT_ROUTE.fullmatch(path):
                report = self.server.store.import_report(match.group(1), match.group(2))
                summary = {key: value for key, value in report.items() if key != "results"}
                self._send_json(200, {"import_batch": summary, "results": report["results"]})
                return
            if match := IMAGE_ROUTE.fullmatch(path):
                query = self._query()
                unknown = set(query) - {"token", "variant", "v"}
                if unknown or len(query.get("variant", [])) != 1 or len(query.get("v", [])) > 1:
                    raise StudioError(400, "INVALID_QUERY", "Exactly one image variant is required.")
                body, media_type, filename = self.server.store.image_bytes(
                    match.group(1), match.group(2), query["variant"][0]
                )
                self._send_bytes(
                    200,
                    body,
                    media_type,
                    disposition=f"inline; filename*=UTF-8''{quote(filename)}",
                    cache_control="private, max-age=31536000, immutable",
                )
                return
            if match := LABEL_ROUTE.fullmatch(path):
                self._send_json(200, {"label": self.server.store.get_label(match.group(1), match.group(2))})
                return
            if match := PRELABEL_ROUTE.fullmatch(path):
                self._send_json(200, {"prelabel": self.server.store.get_prelabel(match.group(1), match.group(2))})
                return
            if match := TRAINING_JOB_ROUTE.fullmatch(path):
                self._send_json(200, {"training_job": self.server.store.training_job(match.group(1), match.group(2))})
                return
            if match := TRAINING_JOBS_ROUTE.fullmatch(path):
                self._send_json(200, {"training_jobs": self.server.store.list_training_jobs(match.group(1))})
                return
            if match := EXPORT_ROUTE.fullmatch(path):
                query = self._query()
                unknown = set(query) - {"token", "mode"}
                if unknown or len(query.get("mode", [])) != 1:
                    raise StudioError(400, "INVALID_QUERY", "Exactly one export mode is required.")
                self._stream_export(match.group(1), query["mode"][0])
                return
            raise StudioError(404, "NOT_FOUND", "Endpoint does not exist.")
        except StudioError as error:
            self.close_connection = True
            self._send_error(error)
        except CLIENT_DISCONNECT_ERRORS:
            pass
        except Exception as exc:
            self.close_connection = True
            self.log_error("Unhandled server error: %r", exc)
            self._send_error(StudioError(500, "INTERNAL_ERROR", "Unexpected local server error."))

    def do_POST(self) -> None:
        try:
            self._validate_host()
            self._validate_framing()
            self._require_token()
            self._require_origin()
            path = unquote(self._parsed().path)
            if path == "/api/v1/projects":
                project = self.server.store.create_project(self._read_json())
                self._send_json(201, {"project": project})
                return
            if match := IMPORTS_ROUTE.fullmatch(path):
                report = self.server.store.create_import(match.group(1), self._read_json())
                report.pop("results", None)
                self._send_json(201, {"import_batch": report})
                return
            if match := EXPORT_ROUTE.fullmatch(path):
                payload = self._read_json()
                if (
                    not isinstance(payload, dict)
                    or "mode" not in payload
                    or set(payload) - {"mode", "sample_ids"}
                ):
                    raise StudioError(
                        422,
                        "INVALID_EXPORT_REQUEST",
                        "Export requires mode and accepts an optional sample_ids array.",
                    )
                self._stream_export(
                    match.group(1),
                    payload["mode"],
                    payload.get("sample_ids"),
                )
                return
            if match := PRELABEL_ROUTE.fullmatch(path):
                payload = self._read_json()
                if not isinstance(payload, dict) or set(payload) != {"layout_id"}:
                    raise StudioError(
                        422,
                        "INVALID_PRELABEL_REQUEST",
                        "Pre-label generation requires exactly one layout_id field.",
                    )
                prelabel = self.server.store.generate_prelabel(
                    match.group(1), match.group(2), payload["layout_id"]
                )
                self._send_json(200, {"prelabel": prelabel})
                return
            if match := ML_FEEDBACK_ROUTE.fullmatch(path):
                payload = self._read_json()
                if not isinstance(payload, dict) or set(payload) != {"expected_revision"}:
                    raise StudioError(
                        422,
                        "INVALID_ML_FEEDBACK_REQUEST",
                        "ML feedback export requires exactly one expected_revision field.",
                    )
                result = self.server.store.export_ml_feedback(
                    match.group(1), match.group(2), payload["expected_revision"]
                )
                self._send_json(200, {"feedback": result})
                return
            if match := TRAINING_JOBS_ROUTE.fullmatch(path):
                job = self.server.store.create_training_job(match.group(1), self._read_json())
                self._send_json(202, {"training_job": job})
                return
            if match := IMPORT_FILE_ROUTE.fullmatch(path):
                query = self._query()
                unknown = set(query) - {"token", "filename"}
                filenames = query.get("filename", [])
                if unknown or len(filenames) != 1:
                    raise StudioError(400, "FILENAME_REQUIRED", "Exactly one filename query value is required.")
                filename = filenames[0]
                self._require_content_type({"application/octet-stream"}, image_wildcard=True)
                length = self._content_length()
                if length > self.server.store.config.max_upload_bytes:
                    error = StudioError(
                        413,
                        "UPLOAD_TOO_LARGE",
                        f"The file exceeds the {self.server.store.config.max_upload_bytes} byte upload limit.",
                    )
                    self.server.store.record_import_failure(
                        match.group(1), match.group(2), filename, error, byte_size=length
                    )
                    self.close_connection = True
                    raise error
                body = self._read_body(self.server.store.config.max_upload_bytes)
                result = self.server.store.import_file(
                    match.group(1), match.group(2), filename, body
                )
                self._send_json(201 if result["result"]["status"] == "imported" else 200, result)
                return
            if match := RECTIFIED_ROUTE.fullmatch(path):
                payload = self._read_json()
                if not isinstance(payload, dict) or set(payload) - {"outer_corners", "width", "height"}:
                    raise StudioError(422, "UNKNOWN_FIELD", "Rectified request contains unknown fields.")
                body = self.server.store.rectified_bytes(
                    match.group(1),
                    match.group(2),
                    payload.get("outer_corners"),
                    payload.get("width"),
                    payload.get("height"),
                )
                self._send_bytes(200, body, "image/png")
                return
            raise StudioError(404, "NOT_FOUND", "Endpoint does not exist.")
        except StudioError as error:
            self.close_connection = True
            self._send_error(error)
        except CLIENT_DISCONNECT_ERRORS:
            pass
        except Exception as exc:
            self.close_connection = True
            self.log_error("Unhandled server error: %r", exc)
            self._send_error(StudioError(500, "INTERNAL_ERROR", "Unexpected local server error."))

    def do_PUT(self) -> None:
        try:
            self._validate_host()
            self._validate_framing()
            self._require_token()
            self._require_origin()
            path = unquote(self._parsed().path)
            match = LABEL_ROUTE.fullmatch(path)
            if not match:
                raise StudioError(404, "NOT_FOUND", "Endpoint does not exist.")
            payload = self._read_json()
            if not isinstance(payload, dict) or set(payload) != {"expected_revision", "label"}:
                raise StudioError(422, "INVALID_LABEL_REQUEST", "Label save requires expected_revision and label.")
            label, unchanged = self.server.store.save_label(
                match.group(1),
                match.group(2),
                payload["label"],
                payload["expected_revision"],
            )
            self._send_json(200, {"label": label, "unchanged": unchanged})
        except StudioError as error:
            self.close_connection = True
            self._send_error(error)
        except CLIENT_DISCONNECT_ERRORS:
            pass
        except Exception as exc:
            self.close_connection = True
            self.log_error("Unhandled server error: %r", exc)
            self._send_error(StudioError(500, "INTERNAL_ERROR", "Unexpected local server error."))

    def do_OPTIONS(self) -> None:
        self._send_error(StudioError(405, "METHOD_NOT_ALLOWED", "Cross-origin requests are not supported."))

    def do_DELETE(self) -> None:
        try:
            self._validate_host()
            self._validate_framing()
            self._require_token()
            self._require_origin()
            path = unquote(self._parsed().path)
            match = SAMPLES_ROUTE.fullmatch(path)
            if not match:
                raise StudioError(404, "NOT_FOUND", "Endpoint does not exist.")
            result = self.server.store.delete_samples(match.group(1), self._read_json())
            self._send_json(200, result)
        except StudioError as error:
            self.close_connection = True
            self._send_error(error)
        except CLIENT_DISCONNECT_ERRORS:
            pass
        except Exception as exc:
            self.close_connection = True
            self.log_error("Unhandled server error: %r", exc)
            self._send_error(StudioError(500, "INTERNAL_ERROR", "Unexpected local server error."))


def build_server(store: StudioStore, host: str, port: int, token: str) -> StudioHTTPServer:
    if host != "127.0.0.1":
        raise ValueError("PTCG Annotation Studio ML only binds to 127.0.0.1")
    if len(token) < 32:
        raise ValueError("The session token must have at least 32 characters")
    return StudioHTTPServer((host, port), store, token)

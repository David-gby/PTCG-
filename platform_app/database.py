from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def parse_json(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return fallback


class PlatformDatabase:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS tenants (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    token_hash TEXT NOT NULL UNIQUE,
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT,
                    expires_at TEXT,
                    last_used_at TEXT
                );
                CREATE TABLE IF NOT EXISTS enterprise_sessions (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL REFERENCES tenants(id),
                    token_hash TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    last_used_at TEXT NOT NULL,
                    revoked_at TEXT
                );
                CREATE INDEX IF NOT EXISTS enterprise_sessions_tenant
                    ON enterprise_sessions(tenant_id, expires_at DESC);
                CREATE TABLE IF NOT EXISTS admin_tokens (
                    id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    token_hash TEXT NOT NULL UNIQUE,
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS admin_users (
                    id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    access_level TEXT NOT NULL CHECK(access_level IN ('owner','annotator')),
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_used_at TEXT,
                    password_changed_at TEXT NOT NULL,
                    login_failed_count INTEGER NOT NULL DEFAULT 0,
                    login_locked_until TEXT
                );
                CREATE TABLE IF NOT EXISTS admin_sessions (
                    id TEXT PRIMARY KEY,
                    admin_user_id TEXT NOT NULL REFERENCES admin_users(id),
                    token_hash TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    last_used_at TEXT NOT NULL,
                    revoked_at TEXT
                );
                CREATE INDEX IF NOT EXISTS admin_sessions_user
                    ON admin_sessions(admin_user_id, expires_at DESC);
                CREATE TABLE IF NOT EXISTS inspections (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL REFERENCES tenants(id),
                    project_id TEXT NOT NULL,
                    sample_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    state TEXT NOT NULL,
                    model_version TEXT,
                    prediction_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    confirmed_at TEXT,
                    feedback_at TEXT
                );
                CREATE INDEX IF NOT EXISTS inspections_tenant_created
                    ON inspections(tenant_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS batch_jobs (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL REFERENCES tenants(id),
                    state TEXT NOT NULL,
                    expected_count INTEGER NOT NULL CHECK(expected_count BETWEEN 1 AND 500),
                    total_bytes INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    last_error TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS batch_jobs_tenant_created
                    ON batch_jobs(tenant_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS batch_items (
                    id TEXT PRIMARY KEY,
                    batch_id TEXT NOT NULL REFERENCES batch_jobs(id),
                    position INTEGER NOT NULL,
                    client_key TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    expected_bytes INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    spool_path TEXT,
                    source_sha256 TEXT,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    inspection_id TEXT REFERENCES inspections(id),
                    error_code TEXT NOT NULL DEFAULT '',
                    error_message TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(batch_id, position),
                    UNIQUE(batch_id, client_key)
                );
                CREATE INDEX IF NOT EXISTS batch_items_queue
                    ON batch_items(state, created_at, position);
                CREATE INDEX IF NOT EXISTS batch_items_batch_position
                    ON batch_items(batch_id, position);
                CREATE TABLE IF NOT EXISTS reference_registration_jobs (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL REFERENCES tenants(id),
                    project_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    capture_filename TEXT NOT NULL,
                    reference_filename TEXT NOT NULL,
                    capture_path TEXT,
                    reference_path TEXT,
                    inspection_id TEXT REFERENCES inspections(id),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS reference_jobs_tenant_created
                    ON reference_registration_jobs(tenant_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    inspection_id TEXT NOT NULL UNIQUE REFERENCES inspections(id),
                    issue_tags_json TEXT NOT NULL,
                    notes TEXT NOT NULL,
                    corrected_inner_json TEXT,
                    corrected_outer_json TEXT,
                    review_status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    reviewed_at TEXT,
                    reviewer TEXT,
                    review_notes TEXT NOT NULL DEFAULT '',
                    exported_feedback_id TEXT
                );
                CREATE INDEX IF NOT EXISTS feedback_status_created
                    ON feedback(review_status, created_at DESC);
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(tenants)").fetchall()
            }
            for name, definition in (
                ("updated_at", "TEXT"),
                ("expires_at", "TEXT"),
                ("last_used_at", "TEXT"),
                ("username", "TEXT"),
                ("password_hash", "TEXT"),
                ("password_changed_at", "TEXT"),
                ("login_failed_count", "INTEGER NOT NULL DEFAULT 0"),
                ("login_locked_until", "TEXT"),
            ):
                if name not in columns:
                    connection.execute(f"ALTER TABLE tenants ADD COLUMN {name} {definition}")
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS tenants_username_unique "
                "ON tenants(username) WHERE username IS NOT NULL"
            )
            feedback_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(feedback)").fetchall()
            }
            if "corrected_outer_json" not in feedback_columns:
                connection.execute("ALTER TABLE feedback ADD COLUMN corrected_outer_json TEXT")

    def empty(self) -> bool:
        with self.connect() as connection:
            tenant_count = int(connection.execute("SELECT COUNT(*) FROM tenants").fetchone()[0])
            admin_count = int(connection.execute("SELECT COUNT(*) FROM admin_tokens").fetchone()[0])
            return tenant_count == 0 and admin_count == 0

    def create_tenant(
        self,
        name: str,
        project_id: str,
        token: str,
        *,
        expires_at: str | None = None,
    ) -> dict[str, Any]:
        tenant_id = f"ten_{secrets.token_hex(8)}"
        created_at = utc_now()
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO tenants(
                       id,name,project_id,token_hash,created_at,updated_at,expires_at
                   ) VALUES(?,?,?,?,?,?,?)""",
                (tenant_id, name, project_id, token_hash(token), created_at, created_at, expires_at),
            )
        return {
            "id": tenant_id,
            "name": name,
            "project_id": project_id,
            "active": True,
            "created_at": created_at,
            "updated_at": created_at,
            "expires_at": expires_at,
            "last_used_at": None,
        }

    def create_admin_token(self, label: str, token: str) -> dict[str, Any]:
        admin_id = f"adm_{secrets.token_hex(8)}"
        created_at = utc_now()
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO admin_tokens(id,label,token_hash,created_at) VALUES(?,?,?,?)",
                (admin_id, label, token_hash(token), created_at),
            )
        return {"id": admin_id, "label": label, "created_at": created_at}

    @staticmethod
    def _admin_user_row(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item.pop("password_hash", None)
        item.pop("login_failed_count", None)
        item.pop("login_locked_until", None)
        item["active"] = bool(item.get("active"))
        return item

    def create_admin_user(
        self,
        label: str,
        username: str,
        password_hash_value: str,
        *,
        access_level: str = "annotator",
    ) -> dict[str, Any]:
        if access_level not in {"owner", "annotator"}:
            raise ValueError(access_level)
        admin_id = f"admusr_{secrets.token_hex(8)}"
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO admin_users(
                       id,label,username,password_hash,access_level,created_at,updated_at,password_changed_at
                   ) VALUES(?,?,?,?,?,?,?,?)""",
                (admin_id, label, username, password_hash_value, access_level, now, now, now),
            )
        result = self.admin_user(admin_id)
        assert result is not None
        return result

    def admin_user(self, admin_user_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM admin_users WHERE id=?", (admin_user_id,)
            ).fetchone()
        return self._admin_user_row(row) if row else None

    def list_admin_users(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM admin_users ORDER BY created_at DESC"
            ).fetchall()
        return [self._admin_user_row(row) for row in rows]

    def admin_user_auth_record(self, username: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT id,label,username,password_hash,access_level,active,created_at,updated_at,
                          last_used_at,password_changed_at,login_failed_count,login_locked_until
                   FROM admin_users WHERE username=?""",
                (username,),
            ).fetchone()
        return dict(row) if row else None

    def set_admin_user_credentials(
        self, admin_user_id: str, username: str, password_hash_value: str
    ) -> dict[str, Any]:
        now = utc_now()
        with self.connect() as connection:
            result = connection.execute(
                """UPDATE admin_users SET username=?,password_hash=?,password_changed_at=?,
                          login_failed_count=0,login_locked_until=NULL,updated_at=? WHERE id=?""",
                (username, password_hash_value, now, now, admin_user_id),
            )
            if result.rowcount != 1:
                raise KeyError(admin_user_id)
            connection.execute(
                "UPDATE admin_sessions SET revoked_at=? WHERE admin_user_id=? AND revoked_at IS NULL",
                (now, admin_user_id),
            )
        result = self.admin_user(admin_user_id)
        assert result is not None
        return result

    def set_admin_user_active(self, admin_user_id: str, active: bool) -> dict[str, Any]:
        now = utc_now()
        with self.connect() as connection:
            result = connection.execute(
                "UPDATE admin_users SET active=?,updated_at=? WHERE id=?",
                (1 if active else 0, now, admin_user_id),
            )
            if result.rowcount != 1:
                raise KeyError(admin_user_id)
            if not active:
                connection.execute(
                    "UPDATE admin_sessions SET revoked_at=? WHERE admin_user_id=? AND revoked_at IS NULL",
                    (now, admin_user_id),
                )
        result = self.admin_user(admin_user_id)
        assert result is not None
        return result

    def delete_admin_user(self, admin_user_id: str) -> bool:
        with self.connect() as connection:
            connection.execute("DELETE FROM admin_sessions WHERE admin_user_id=?", (admin_user_id,))
            result = connection.execute("DELETE FROM admin_users WHERE id=?", (admin_user_id,))
        return result.rowcount == 1

    def record_admin_login_failure(
        self, admin_user_id: str, *, lock_after: int, lock_seconds: int
    ) -> str | None:
        now_dt = datetime.now(timezone.utc).replace(microsecond=0)
        now = now_dt.isoformat().replace("+00:00", "Z")
        with self.connect() as connection:
            row = connection.execute(
                "SELECT login_failed_count,login_locked_until FROM admin_users WHERE id=?",
                (admin_user_id,),
            ).fetchone()
            if row is None:
                return None
            locked_until = str(row["login_locked_until"] or "") or None
            failed_count = 0 if locked_until and locked_until <= now else int(row["login_failed_count"] or 0)
            failed_count += 1
            if failed_count >= lock_after:
                locked_until = datetime.fromtimestamp(
                    now_dt.timestamp() + lock_seconds, timezone.utc
                ).isoformat().replace("+00:00", "Z")
                failed_count = 0
            connection.execute(
                """UPDATE admin_users SET login_failed_count=?,login_locked_until=?,updated_at=?
                   WHERE id=?""",
                (failed_count, locked_until, now, admin_user_id),
            )
        return locked_until

    def record_admin_login_success(self, admin_user_id: str) -> None:
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                """UPDATE admin_users SET login_failed_count=0,login_locked_until=NULL,
                          last_used_at=?,updated_at=? WHERE id=?""",
                (now, now, admin_user_id),
            )

    def create_admin_session(self, admin_user_id: str, token: str, expires_at: str) -> None:
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO admin_sessions(
                       id,admin_user_id,token_hash,created_at,expires_at,last_used_at
                   ) VALUES(?,?,?,?,?,?)""",
                (f"ases_{secrets.token_hex(10)}", admin_user_id, token_hash(token), now, expires_at, now),
            )
            connection.execute(
                "DELETE FROM admin_sessions WHERE expires_at<? OR revoked_at IS NOT NULL", (now,)
            )

    def authenticate(self, token: str) -> dict[str, Any] | None:
        if not token or len(token) > 256:
            return None
        digest = token_hash(token)
        now = utc_now()
        with self.connect() as connection:
            tenant = connection.execute(
                """SELECT id,name,project_id,created_at,updated_at,expires_at,last_used_at
                   FROM tenants WHERE token_hash=? AND active=1
                   AND (expires_at IS NULL OR expires_at>?)""",
                (digest, now),
            ).fetchone()
            if tenant:
                connection.execute(
                    "UPDATE tenants SET last_used_at=? WHERE id=?", (now, str(tenant["id"]))
                )
                return {"role": "enterprise", "tenant": dict(tenant)}
            admin = connection.execute(
                "SELECT id,label,created_at FROM admin_tokens WHERE token_hash=? AND active=1",
                (digest,),
            ).fetchone()
            if admin:
                item = dict(admin)
                item.update({"access_level": "owner", "auth_type": "token", "username": None})
                return {"role": "admin", "admin": item}
        return None

    def tenant_auth_record(self, username: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT id,name,project_id,active,created_at,updated_at,expires_at,
                          last_used_at,username,password_hash,password_changed_at,
                          login_failed_count,login_locked_until
                   FROM tenants WHERE username=?""",
                (username,),
            ).fetchone()
        return dict(row) if row else None

    def set_tenant_credentials(
        self,
        tenant_id: str,
        username: str,
        password_hash_value: str,
    ) -> dict[str, Any]:
        now = utc_now()
        with self.connect() as connection:
            result = connection.execute(
                """UPDATE tenants SET username=?,password_hash=?,password_changed_at=?,
                          login_failed_count=0,login_locked_until=NULL,updated_at=?
                   WHERE id=?""",
                (username, password_hash_value, now, now, tenant_id),
            )
            if result.rowcount != 1:
                raise KeyError(tenant_id)
            connection.execute(
                "UPDATE enterprise_sessions SET revoked_at=? WHERE tenant_id=? AND revoked_at IS NULL",
                (now, tenant_id),
            )
        tenant = self.tenant(tenant_id)
        assert tenant is not None
        return tenant

    def record_login_failure(self, tenant_id: str, *, lock_after: int, lock_seconds: int) -> str | None:
        now_dt = datetime.now(timezone.utc).replace(microsecond=0)
        now = now_dt.isoformat().replace("+00:00", "Z")
        with self.connect() as connection:
            row = connection.execute(
                "SELECT login_failed_count,login_locked_until FROM tenants WHERE id=?",
                (tenant_id,),
            ).fetchone()
            if row is None:
                return None
            locked_until = str(row["login_locked_until"] or "") or None
            if locked_until and locked_until <= now:
                failed_count = 0
                locked_until = None
            else:
                failed_count = int(row["login_failed_count"] or 0)
            failed_count += 1
            if failed_count >= lock_after:
                locked_until = (
                    now_dt.timestamp() + lock_seconds
                )
                locked_until = datetime.fromtimestamp(locked_until, timezone.utc).isoformat().replace("+00:00", "Z")
                failed_count = 0
            connection.execute(
                "UPDATE tenants SET login_failed_count=?,login_locked_until=?,updated_at=? WHERE id=?",
                (failed_count, locked_until, now, tenant_id),
            )
        return locked_until

    def record_login_success(self, tenant_id: str) -> None:
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                """UPDATE tenants SET login_failed_count=0,login_locked_until=NULL,
                          last_used_at=?,updated_at=? WHERE id=?""",
                (now, now, tenant_id),
            )

    def create_enterprise_session(self, tenant_id: str, token: str, expires_at: str) -> None:
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO enterprise_sessions(
                       id,tenant_id,token_hash,created_at,expires_at,last_used_at
                   ) VALUES(?,?,?,?,?,?)""",
                (f"ses_{secrets.token_hex(10)}", tenant_id, token_hash(token), now, expires_at, now),
            )
            connection.execute(
                "DELETE FROM enterprise_sessions WHERE expires_at<? OR revoked_at IS NOT NULL",
                (now,),
            )

    def authenticate_session(self, token: str) -> dict[str, Any] | None:
        if not token or len(token) > 256:
            return None
        digest = token_hash(token)
        now = utc_now()
        with self.connect() as connection:
            row = connection.execute(
                """SELECT t.id,t.name,t.project_id,t.created_at,t.updated_at,
                          t.expires_at,t.last_used_at,s.id AS session_id,s.expires_at AS session_expires_at
                   FROM enterprise_sessions s
                   JOIN tenants t ON t.id=s.tenant_id
                   WHERE s.token_hash=? AND s.revoked_at IS NULL AND s.expires_at>?
                     AND t.active=1 AND (t.expires_at IS NULL OR t.expires_at>?)""",
                (digest, now, now),
            ).fetchone()
            if row:
                connection.execute(
                    "UPDATE enterprise_sessions SET last_used_at=? WHERE id=?",
                    (now, str(row["session_id"])),
                )
                connection.execute(
                    "UPDATE tenants SET last_used_at=? WHERE id=?",
                    (now, str(row["id"])),
                )
                tenant = dict(row)
                tenant.pop("session_id", None)
                session_expires_at = tenant.pop("session_expires_at", None)
                return {
                    "role": "enterprise",
                    "tenant": tenant,
                    "session_expires_at": session_expires_at,
                }
            admin = connection.execute(
                """SELECT u.id,u.label,u.username,u.access_level,u.created_at,u.updated_at,
                          u.last_used_at,s.id AS session_id,s.expires_at AS session_expires_at
                   FROM admin_sessions s JOIN admin_users u ON u.id=s.admin_user_id
                   WHERE s.token_hash=? AND s.revoked_at IS NULL AND s.expires_at>?
                     AND u.active=1""",
                (digest, now),
            ).fetchone()
            if not admin:
                return None
            connection.execute(
                "UPDATE admin_sessions SET last_used_at=? WHERE id=?",
                (now, str(admin["session_id"])),
            )
            connection.execute(
                "UPDATE admin_users SET last_used_at=?,updated_at=? WHERE id=?",
                (now, now, str(admin["id"])),
            )
            item = dict(admin)
            item.pop("session_id", None)
            session_expires_at = item.pop("session_expires_at", None)
            item["auth_type"] = "password"
            return {
                "role": "admin",
                "admin": item,
                "session_expires_at": session_expires_at,
            }

    def revoke_enterprise_session(self, token: str) -> None:
        if not token or len(token) > 256:
            return
        with self.connect() as connection:
            connection.execute(
                "UPDATE enterprise_sessions SET revoked_at=? WHERE token_hash=? AND revoked_at IS NULL",
                (utc_now(), token_hash(token)),
            )

    def revoke_admin_session(self, token: str) -> None:
        if not token or len(token) > 256:
            return
        with self.connect() as connection:
            connection.execute(
                "UPDATE admin_sessions SET revoked_at=? WHERE token_hash=? AND revoked_at IS NULL",
                (utc_now(), token_hash(token)),
            )

    def revoke_session(self, token: str) -> None:
        self.revoke_enterprise_session(token)
        self.revoke_admin_session(token)

    def revoke_tenant_sessions(self, tenant_id: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE enterprise_sessions SET revoked_at=? WHERE tenant_id=? AND revoked_at IS NULL",
                (utc_now(), tenant_id),
            )

    @staticmethod
    def _tenant_row(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item.pop("token_hash", None)
        password_configured = bool(item.get("password_hash"))
        item.pop("password_hash", None)
        item.pop("login_failed_count", None)
        item.pop("login_locked_until", None)
        item["active"] = bool(item.get("active"))
        item["account_configured"] = bool(item.get("username") and password_configured)
        item["inspection_count"] = int(item.get("inspection_count") or 0)
        item["pending_feedback_count"] = int(item.get("pending_feedback_count") or 0)
        return item

    def tenant(self, tenant_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT t.*,
                          COUNT(DISTINCT i.id) AS inspection_count,
                          COUNT(DISTINCT CASE WHEN f.review_status='pending' THEN f.id END) AS pending_feedback_count
                   FROM tenants t
                   LEFT JOIN inspections i ON i.tenant_id=t.id
                   LEFT JOIN feedback f ON f.inspection_id=i.id
                   WHERE t.id=? GROUP BY t.id""",
                (tenant_id,),
            ).fetchone()
        return self._tenant_row(row) if row else None

    def list_tenants(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT t.*,
                          COUNT(DISTINCT i.id) AS inspection_count,
                          COUNT(DISTINCT CASE WHEN f.review_status='pending' THEN f.id END) AS pending_feedback_count
                   FROM tenants t
                   LEFT JOIN inspections i ON i.tenant_id=t.id
                   LEFT JOIN feedback f ON f.inspection_id=i.id
                   GROUP BY t.id ORDER BY t.created_at DESC"""
            ).fetchall()
        return [self._tenant_row(row) for row in rows]

    def rotate_tenant_token(self, tenant_id: str, token: str) -> dict[str, Any]:
        now = utc_now()
        with self.connect() as connection:
            result = connection.execute(
                "UPDATE tenants SET token_hash=?,updated_at=? WHERE id=?",
                (token_hash(token), now, tenant_id),
            )
            if result.rowcount != 1:
                raise KeyError(tenant_id)
        tenant = self.tenant(tenant_id)
        assert tenant is not None
        return tenant

    def set_tenant_active(self, tenant_id: str, active: bool) -> dict[str, Any]:
        now = utc_now()
        with self.connect() as connection:
            result = connection.execute(
                "UPDATE tenants SET active=?,updated_at=? WHERE id=?",
                (1 if active else 0, now, tenant_id),
            )
            if result.rowcount != 1:
                raise KeyError(tenant_id)
            if not active:
                connection.execute(
                    "UPDATE enterprise_sessions SET revoked_at=? WHERE tenant_id=? AND revoked_at IS NULL",
                    (now, tenant_id),
                )
        tenant = self.tenant(tenant_id)
        assert tenant is not None
        return tenant

    def set_tenant_expiry(self, tenant_id: str, expires_at: str | None) -> dict[str, Any]:
        now = utc_now()
        with self.connect() as connection:
            result = connection.execute(
                "UPDATE tenants SET expires_at=?,updated_at=? WHERE id=?",
                (expires_at, now, tenant_id),
            )
            if result.rowcount != 1:
                raise KeyError(tenant_id)
        tenant = self.tenant(tenant_id)
        assert tenant is not None
        return tenant

    def tenant_deletion_manifest(self, tenant_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            tenant = connection.execute(
                "SELECT id,name,project_id FROM tenants WHERE id=?", (tenant_id,)
            ).fetchone()
            if tenant is None:
                return None
            inspections = connection.execute(
                "SELECT id,sample_id FROM inspections WHERE tenant_id=? ORDER BY created_at",
                (tenant_id,),
            ).fetchall()
            feedback = connection.execute(
                """SELECT f.id,f.exported_feedback_id,f.review_status,i.sample_id
                   FROM feedback f JOIN inspections i ON i.id=f.inspection_id
                   WHERE i.tenant_id=? ORDER BY f.created_at""",
                (tenant_id,),
            ).fetchall()
        result = dict(tenant)
        result["inspections"] = [dict(row) for row in inspections]
        result["feedback"] = [dict(row) for row in feedback]
        return result

    def delete_tenant(self, tenant_id: str) -> dict[str, int]:
        with self.connect() as connection:
            batch_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM batch_jobs WHERE tenant_id=?", (tenant_id,)
                ).fetchone()[0]
            )
            feedback_count = int(
                connection.execute(
                    """SELECT COUNT(*) FROM feedback f JOIN inspections i ON i.id=f.inspection_id
                       WHERE i.tenant_id=?""",
                    (tenant_id,),
                ).fetchone()[0]
            )
            inspection_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM inspections WHERE tenant_id=?", (tenant_id,)
                ).fetchone()[0]
            )
            connection.execute(
                "DELETE FROM batch_items WHERE batch_id IN (SELECT id FROM batch_jobs WHERE tenant_id=?)",
                (tenant_id,),
            )
            connection.execute("DELETE FROM batch_jobs WHERE tenant_id=?", (tenant_id,))
            connection.execute("DELETE FROM reference_registration_jobs WHERE tenant_id=?", (tenant_id,))
            connection.execute(
                "DELETE FROM feedback WHERE inspection_id IN (SELECT id FROM inspections WHERE tenant_id=?)",
                (tenant_id,),
            )
            connection.execute("DELETE FROM inspections WHERE tenant_id=?", (tenant_id,))
            connection.execute("DELETE FROM enterprise_sessions WHERE tenant_id=?", (tenant_id,))
            result = connection.execute("DELETE FROM tenants WHERE id=?", (tenant_id,))
            if result.rowcount != 1:
                raise KeyError(tenant_id)
        return {
            "inspections": inspection_count,
            "feedback": feedback_count,
            "batches": batch_count,
        }

    def create_batch(
        self,
        tenant_id: str,
        items: list[dict[str, Any]],
        total_bytes: int,
    ) -> str:
        batch_id = f"bat_{secrets.token_hex(10)}"
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO batch_jobs(
                       id,tenant_id,state,expected_count,total_bytes,created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,?)""",
                (batch_id, tenant_id, "uploading", len(items), int(total_bytes), now, now),
            )
            for position, item in enumerate(items):
                connection.execute(
                    """INSERT INTO batch_items(
                           id,batch_id,position,client_key,filename,content_type,
                           expected_bytes,state,created_at,updated_at
                       ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (
                        f"bti_{secrets.token_hex(10)}",
                        batch_id,
                        position,
                        item["client_key"],
                        item["filename"],
                        item["content_type"],
                        int(item["size"]),
                        "waiting_upload",
                        now,
                        now,
                    ),
                )
        return batch_id

    @staticmethod
    def _batch_item_row(row: sqlite3.Row) -> dict[str, Any]:
        return dict(row)

    def _batch_counts(
        self, connection: sqlite3.Connection, batch_id: str
    ) -> dict[str, int]:
        rows = connection.execute(
            "SELECT state,COUNT(*) AS amount FROM batch_items WHERE batch_id=? GROUP BY state",
            (batch_id,),
        ).fetchall()
        counts = {
            "waiting_upload": 0,
            "queued": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
        }
        for row in rows:
            counts[str(row["state"])] = int(row["amount"])
        return counts

    def _refresh_batch_state(
        self, connection: sqlite3.Connection, batch_id: str
    ) -> None:
        row = connection.execute(
            "SELECT state,expected_count,started_at FROM batch_jobs WHERE id=?",
            (batch_id,),
        ).fetchone()
        if row is None or str(row["state"]) in {"paused", "cancelled"}:
            return
        counts = self._batch_counts(connection, batch_id)
        expected = int(row["expected_count"])
        terminal = counts["completed"] + counts["failed"] + counts["cancelled"]
        if counts["waiting_upload"]:
            state = "uploading"
        elif counts["processing"]:
            state = "processing"
        elif counts["queued"]:
            state = "queued"
        elif terminal >= expected and counts["failed"] and counts["completed"]:
            state = "partial"
        elif terminal >= expected and counts["failed"]:
            state = "failed"
        elif terminal >= expected:
            state = "completed"
        else:
            state = "uploading"
        now = utc_now()
        completed_at = now if state in {"completed", "partial", "failed"} else None
        connection.execute(
            """UPDATE batch_jobs SET state=?,updated_at=?,
                   started_at=CASE WHEN ? IN ('processing','completed','partial','failed')
                                   THEN COALESCE(started_at,?) ELSE started_at END,
                   completed_at=? WHERE id=?""",
            (state, now, state, now, completed_at, batch_id),
        )

    def batch(self, batch_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT b.*,t.name AS tenant_name
                   FROM batch_jobs b LEFT JOIN tenants t ON t.id=b.tenant_id
                   WHERE b.id=?""",
                (batch_id,),
            ).fetchone()
            if row is None:
                return None
            counts = self._batch_counts(connection, batch_id)
            items = connection.execute(
                "SELECT * FROM batch_items WHERE batch_id=? ORDER BY position",
                (batch_id,),
            ).fetchall()
        result = dict(row)
        result["counts"] = counts
        result["items"] = [self._batch_item_row(item) for item in items]
        return result

    def list_batches(self, tenant_id: str, limit: int = 20) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 100))
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT id FROM batch_jobs WHERE tenant_id=?
                   ORDER BY created_at DESC LIMIT ?""",
                (tenant_id, limit),
            ).fetchall()
        return [
            batch
            for row in rows
            if (batch := self.batch(str(row["id"]))) is not None
        ]

    def batch_ids_for_tenant(self, tenant_id: str) -> list[str]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT id FROM batch_jobs WHERE tenant_id=?",
                (tenant_id,),
            ).fetchall()
        return [str(row["id"]) for row in rows]

    def batch_item(self, batch_id: str, item_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM batch_items WHERE batch_id=? AND id=?",
                (batch_id, item_id),
            ).fetchone()
        return self._batch_item_row(row) if row else None

    def mark_batch_item_uploaded(
        self,
        batch_id: str,
        item_id: str,
        *,
        spool_path: str,
        source_sha256: str,
    ) -> dict[str, Any]:
        now = utc_now()
        with self.connect() as connection:
            result = connection.execute(
                """UPDATE batch_items SET state='queued',spool_path=?,source_sha256=?,
                       error_code='',error_message='',updated_at=?
                   WHERE batch_id=? AND id=? AND state='waiting_upload'""",
                (spool_path, source_sha256, now, batch_id, item_id),
            )
            if result.rowcount != 1:
                row = connection.execute(
                    "SELECT * FROM batch_items WHERE batch_id=? AND id=?",
                    (batch_id, item_id),
                ).fetchone()
                if row is None:
                    raise KeyError(item_id)
                return self._batch_item_row(row)
            self._refresh_batch_state(connection, batch_id)
            row = connection.execute(
                "SELECT * FROM batch_items WHERE id=?", (item_id,)
            ).fetchone()
        return self._batch_item_row(row)

    def recover_batch_queue(self) -> int:
        now = utc_now()
        with self.connect() as connection:
            result = connection.execute(
                """UPDATE batch_items SET state='queued',updated_at=?,
                       error_code='SERVER_RESTARTED',
                       error_message='平台重启后已自动重新排队。'
                   WHERE state='processing'""",
                (now,),
            )
            connection.execute(
                """UPDATE batch_jobs SET state='queued',updated_at=?,completed_at=NULL
                   WHERE state='processing'""",
                (now,),
            )
        return int(result.rowcount)

    def claim_next_batch_item(self) -> dict[str, Any] | None:
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT i.*,b.tenant_id
                   FROM batch_items i JOIN batch_jobs b ON b.id=i.batch_id
                   WHERE i.state='queued' AND b.state NOT IN ('paused','cancelled')
                   ORDER BY i.created_at,i.position LIMIT 1"""
            ).fetchone()
            if row is None:
                return None
            result = connection.execute(
                """UPDATE batch_items SET state='processing',attempt_count=attempt_count+1,
                       updated_at=? WHERE id=? AND state='queued'""",
                (now, row["id"]),
            )
            if result.rowcount != 1:
                return None
            connection.execute(
                """UPDATE batch_jobs SET state='processing',started_at=COALESCE(started_at,?),
                       updated_at=? WHERE id=?""",
                (now, now, row["batch_id"]),
            )
            claimed = connection.execute(
                """SELECT i.*,b.tenant_id
                   FROM batch_items i JOIN batch_jobs b ON b.id=i.batch_id
                   WHERE i.id=?""",
                (row["id"],),
            ).fetchone()
        return self._batch_item_row(claimed)

    def finish_batch_item(
        self,
        batch_id: str,
        item_id: str,
        *,
        inspection_id: str | None = None,
        error_code: str = "",
        error_message: str = "",
    ) -> None:
        now = utc_now()
        state = "completed" if inspection_id else "failed"
        with self.connect() as connection:
            connection.execute(
                """UPDATE batch_items SET state=?,inspection_id=?,error_code=?,
                       error_message=?,updated_at=? WHERE batch_id=? AND id=?""",
                (
                    state,
                    inspection_id,
                    error_code[:128],
                    error_message[:1000],
                    now,
                    batch_id,
                    item_id,
                ),
            )
            if error_message:
                connection.execute(
                    "UPDATE batch_jobs SET last_error=?,updated_at=? WHERE id=?",
                    (error_message[:1000], now, batch_id),
                )
            self._refresh_batch_state(connection, batch_id)

    def batch_action(self, batch_id: str, action: str) -> None:
        now = utc_now()
        with self.connect() as connection:
            row = connection.execute(
                "SELECT state FROM batch_jobs WHERE id=?", (batch_id,)
            ).fetchone()
            if row is None:
                raise KeyError(batch_id)
            state = str(row["state"])
            if action == "pause":
                if state not in {"completed", "partial", "failed", "cancelled"}:
                    connection.execute(
                        "UPDATE batch_jobs SET state='paused',updated_at=? WHERE id=?",
                        (now, batch_id),
                    )
            elif action == "resume":
                if state == "paused":
                    connection.execute(
                        "UPDATE batch_jobs SET state='queued',updated_at=? WHERE id=?",
                        (now, batch_id),
                    )
                    self._refresh_batch_state(connection, batch_id)
            elif action == "retry":
                connection.execute(
                    """UPDATE batch_items SET state='queued',error_code='',error_message='',
                           updated_at=? WHERE batch_id=? AND state='failed' AND spool_path IS NOT NULL""",
                    (now, batch_id),
                )
                connection.execute(
                    "UPDATE batch_jobs SET state='queued',completed_at=NULL,last_error='',updated_at=? WHERE id=?",
                    (now, batch_id),
                )
                self._refresh_batch_state(connection, batch_id)
            elif action == "cancel":
                connection.execute(
                    """UPDATE batch_items SET state='cancelled',updated_at=?
                       WHERE batch_id=? AND state IN ('waiting_upload','queued','failed')""",
                    (now, batch_id),
                )
                connection.execute(
                    """UPDATE batch_jobs SET state='cancelled',updated_at=?,completed_at=?
                       WHERE id=?""",
                    (now, now, batch_id),
                )
            else:
                raise ValueError(action)

    def create_reference_job(
        self, tenant_id: str, project_id: str, capture_filename: str, reference_filename: str
    ) -> str:
        job_id = f"reg_{secrets.token_hex(10)}"
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO reference_registration_jobs(
                    id,tenant_id,project_id,state,capture_filename,reference_filename,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?)""",
                (job_id, tenant_id, project_id, "waiting_capture", capture_filename, reference_filename, now, now),
            )
        return job_id

    def reference_job(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM reference_registration_jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def mark_reference_file_uploaded(self, job_id: str, kind: str, path: str) -> dict[str, Any]:
        column = {"capture": "capture_path", "reference": "reference_path"}.get(kind)
        if column is None:
            raise ValueError(kind)
        with self.connect() as connection:
            connection.execute(f"UPDATE reference_registration_jobs SET {column}=?,updated_at=? WHERE id=?", (path, utc_now(), job_id))
            row = connection.execute("SELECT * FROM reference_registration_jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            state = "ready" if row["capture_path"] and row["reference_path"] else ("waiting_reference" if row["capture_path"] else "waiting_capture")
            connection.execute("UPDATE reference_registration_jobs SET state=?,updated_at=? WHERE id=?", (state, utc_now(), job_id))
            row = connection.execute("SELECT * FROM reference_registration_jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row)

    def finish_reference_job(self, job_id: str, inspection_id: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE reference_registration_jobs SET state='completed',inspection_id=?,updated_at=? WHERE id=?",
                (inspection_id, utc_now(), job_id),
            )

    def add_inspection(
        self,
        *,
        tenant_id: str,
        project_id: str,
        sample_id: str,
        filename: str,
        state: str,
        model_version: str | None,
        prediction: dict[str, Any],
    ) -> str:
        inspection_id = f"ins_{secrets.token_hex(10)}"
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO inspections(
                       id,tenant_id,project_id,sample_id,filename,state,model_version,
                       prediction_json,created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    inspection_id,
                    tenant_id,
                    project_id,
                    sample_id,
                    filename,
                    state,
                    model_version,
                    json_text(prediction),
                    now,
                    now,
                ),
            )
        return inspection_id

    @staticmethod
    def _inspection_row(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["prediction"] = parse_json(item.pop("prediction_json", ""), {})
        return item

    def inspection(self, inspection_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT i.*,t.name AS tenant_name
                   FROM inspections i LEFT JOIN tenants t ON t.id=i.tenant_id
                   WHERE i.id=?""",
                (inspection_id,),
            ).fetchone()
        return self._inspection_row(row) if row else None

    @staticmethod
    def _inspection_state_clause(states: list[str] | tuple[str, ...] | None) -> tuple[str, list[str]]:
        normalized = [str(state) for state in (states or []) if str(state)]
        if not normalized:
            return "", []
        placeholders = ",".join("?" for _ in normalized)
        return f" AND i.state IN ({placeholders})", normalized

    def list_inspections(
        self,
        tenant_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
        states: list[str] | tuple[str, ...] | None = None,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 500))
        offset = max(0, int(offset))
        state_clause, state_params = self._inspection_state_clause(states)
        with self.connect() as connection:
            if tenant_id:
                rows = connection.execute(
                    f"""SELECT i.*,t.name AS tenant_name
                       FROM inspections i LEFT JOIN tenants t ON t.id=i.tenant_id
                       WHERE i.tenant_id=?{state_clause}
                       ORDER BY i.created_at DESC,i.id DESC LIMIT ? OFFSET ?""",
                    (tenant_id, *state_params, limit, offset),
                ).fetchall()
            else:
                rows = connection.execute(
                    f"""SELECT i.*,t.name AS tenant_name
                       FROM inspections i LEFT JOIN tenants t ON t.id=i.tenant_id
                       WHERE 1=1{state_clause}
                       ORDER BY i.created_at DESC,i.id DESC LIMIT ? OFFSET ?""",
                    (*state_params, limit, offset),
                ).fetchall()
        return [self._inspection_row(row) for row in rows]

    def count_inspections(
        self,
        tenant_id: str | None = None,
        states: list[str] | tuple[str, ...] | None = None,
    ) -> int:
        state_clause, state_params = self._inspection_state_clause(states)
        with self.connect() as connection:
            if tenant_id:
                row = connection.execute(
                    f"SELECT COUNT(*) FROM inspections i WHERE i.tenant_id=?{state_clause}",
                    (tenant_id, *state_params),
                ).fetchone()
            else:
                row = connection.execute(
                    f"SELECT COUNT(*) FROM inspections i WHERE 1=1{state_clause}",
                    tuple(state_params),
                ).fetchone()
        return int(row[0]) if row else 0

    def mark_confirmed(self, inspection_id: str) -> None:
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                "UPDATE inspections SET state='confirmed',confirmed_at=?,updated_at=? WHERE id=?",
                (now, now, inspection_id),
            )

    def submit_feedback(
        self,
        inspection_id: str,
        issue_tags: list[str],
        notes: str,
        corrected_inner: dict[str, float] | None,
        corrected_outer: list[list[float]] | None,
    ) -> dict[str, Any]:
        now = utc_now()
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT id FROM feedback WHERE inspection_id=?", (inspection_id,)
            ).fetchone()
            if existing:
                feedback_id = str(existing["id"])
                connection.execute(
                    """UPDATE feedback SET issue_tags_json=?,notes=?,corrected_inner_json=?,corrected_outer_json=?,
                           review_status='pending',updated_at=?,reviewed_at=NULL,reviewer=NULL,
                           review_notes='',exported_feedback_id=NULL WHERE id=?""",
                    (
                        json_text(issue_tags),
                        notes,
                        json_text(corrected_inner) if corrected_inner else None,
                        json_text(corrected_outer) if corrected_outer else None,
                        now,
                        feedback_id,
                    ),
                )
            else:
                feedback_id = f"fbk_{secrets.token_hex(10)}"
                connection.execute(
                    """INSERT INTO feedback(
                           id,inspection_id,issue_tags_json,notes,corrected_inner_json,corrected_outer_json,
                           review_status,created_at,updated_at
                       ) VALUES(?,?,?,?,?,?,'pending',?,?)""",
                    (
                        feedback_id,
                        inspection_id,
                        json_text(issue_tags),
                        notes,
                        json_text(corrected_inner) if corrected_inner else None,
                        json_text(corrected_outer) if corrected_outer else None,
                        now,
                        now,
                    ),
                )
            connection.execute(
                "UPDATE inspections SET state='feedback_pending',feedback_at=?,updated_at=? WHERE id=?",
                (now, now, inspection_id),
            )
        return self.feedback(feedback_id) or {}

    @staticmethod
    def _feedback_row(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["issue_tags"] = parse_json(item.pop("issue_tags_json", ""), [])
        item["corrected_inner"] = parse_json(item.pop("corrected_inner_json", None), None)
        item["corrected_outer"] = parse_json(item.pop("corrected_outer_json", None), None)
        if "prediction_json" in item:
            item["prediction"] = parse_json(item.pop("prediction_json", ""), {})
        return item

    def feedback(self, feedback_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT f.*,i.tenant_id,i.project_id,i.sample_id,i.filename,i.model_version,
                          i.prediction_json,i.state AS inspection_state,t.name AS tenant_name
                   FROM feedback f JOIN inspections i ON i.id=f.inspection_id
                   JOIN tenants t ON t.id=i.tenant_id WHERE f.id=?""",
                (feedback_id,),
            ).fetchone()
        return self._feedback_row(row) if row else None

    def feedback_for_inspection(self, inspection_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM feedback WHERE inspection_id=?", (inspection_id,)
            ).fetchone()
        return self._feedback_row(row) if row else None

    def delete_feedback_inspection(self, feedback_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT f.id AS feedback_id,f.exported_feedback_id,f.review_status,
                          i.id AS inspection_id,i.tenant_id,i.project_id,i.sample_id,i.filename
                   FROM feedback f JOIN inspections i ON i.id=f.inspection_id WHERE f.id=?""",
                (feedback_id,),
            ).fetchone()
            if row is None:
                raise KeyError(feedback_id)
            result = dict(row)
            connection.execute("DELETE FROM feedback WHERE id=?", (feedback_id,))
            # Batch/reference audit rows may still point at the inspection. Keep
            # those upload records, but detach the target before deleting it.
            connection.execute(
                "UPDATE batch_items SET inspection_id=NULL WHERE inspection_id=?",
                (str(row["inspection_id"]),),
            )
            connection.execute(
                "UPDATE reference_registration_jobs SET inspection_id=NULL WHERE inspection_id=?",
                (str(row["inspection_id"]),),
            )
            connection.execute("DELETE FROM inspections WHERE id=?", (str(row["inspection_id"]),))
        return result

    def list_feedback(self, status: str | None = None, limit: int = 300) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 1000))
        query = """SELECT f.*,i.tenant_id,i.project_id,i.sample_id,i.filename,i.model_version,
                          i.prediction_json,i.state AS inspection_state,t.name AS tenant_name
                   FROM feedback f JOIN inspections i ON i.id=f.inspection_id
                   JOIN tenants t ON t.id=i.tenant_id"""
        params: tuple[Any, ...]
        if status:
            query += " WHERE f.review_status=?"
            params = (status, limit)
        else:
            params = (limit,)
        query += " ORDER BY f.created_at DESC LIMIT ?"
        with self.connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._feedback_row(row) for row in rows]

    def list_training_feedback(self) -> list[dict[str, Any]]:
        """Return the complete approved training pool without the UI's 1000-row cap."""
        query = """SELECT f.*,i.tenant_id,i.project_id,i.sample_id,i.filename,i.model_version,
                          i.prediction_json,i.state AS inspection_state,t.name AS tenant_name
                   FROM feedback f JOIN inspections i ON i.id=f.inspection_id
                   JOIN tenants t ON t.id=i.tenant_id
                   WHERE f.review_status='approved' AND f.exported_feedback_id IS NOT NULL
                   ORDER BY f.created_at"""
        with self.connect() as connection:
            rows = connection.execute(query).fetchall()
        return [self._feedback_row(row) for row in rows]

    def review_feedback(
        self,
        feedback_id: str,
        *,
        status: str,
        reviewer: str,
        notes: str,
        corrected_inner: dict[str, float] | None,
        corrected_outer: list[list[float]] | None,
        exported_feedback_id: str | None = None,
    ) -> None:
        now = utc_now()
        inspection_state = {
            "approved": "feedback_approved",
            "needs_annotation": "feedback_needs_annotation",
            "discarded": "feedback_discarded",
            "rejected": "feedback_rejected",
        }[status]
        with self.connect() as connection:
            row = connection.execute(
                "SELECT inspection_id FROM feedback WHERE id=?", (feedback_id,)
            ).fetchone()
            if not row:
                raise KeyError(feedback_id)
            connection.execute(
                """UPDATE feedback SET review_status=?,reviewer=?,review_notes=?,
                       corrected_inner_json=?,corrected_outer_json=?,reviewed_at=?,updated_at=?,exported_feedback_id=?
                   WHERE id=?""",
                (
                    status,
                    reviewer,
                    notes,
                    json_text(corrected_inner) if corrected_inner else None,
                    json_text(corrected_outer) if corrected_outer else None,
                    now,
                    now,
                    exported_feedback_id,
                    feedback_id,
                ),
            )
            connection.execute(
                "UPDATE inspections SET state=?,updated_at=? WHERE id=?",
                (inspection_state, now, str(row["inspection_id"])),
            )

    def reopen_feedback(
        self,
        feedback_id: str,
        *,
        reviewer: str,
        notes: str,
    ) -> None:
        """Return an approved item to the pending queue without discarding its draft geometry."""
        now = utc_now()
        with self.connect() as connection:
            row = connection.execute(
                "SELECT inspection_id,review_status FROM feedback WHERE id=?", (feedback_id,)
            ).fetchone()
            if not row:
                raise KeyError(feedback_id)
            if str(row["review_status"]) != "approved":
                raise ValueError("Only approved feedback can be reopened.")
            connection.execute(
                """UPDATE feedback SET review_status='pending',reviewer=?,review_notes=?,
                       reviewed_at=?,updated_at=?,exported_feedback_id=NULL WHERE id=?""",
                (reviewer, notes, now, now, feedback_id),
            )
            connection.execute(
                "UPDATE inspections SET state='feedback_pending',updated_at=? WHERE id=?",
                (now, str(row["inspection_id"])),
            )

    def summary(self) -> dict[str, int]:
        with self.connect() as connection:
            tenants_active = connection.execute(
                """SELECT COUNT(*) FROM tenants WHERE active=1
                   AND (expires_at IS NULL OR expires_at>?)""",
                (utc_now(),),
            ).fetchone()[0]
            inspections = connection.execute("SELECT COUNT(*) FROM inspections").fetchone()[0]
            confirmed = connection.execute(
                "SELECT COUNT(*) FROM inspections WHERE state='confirmed'"
            ).fetchone()[0]
            pending = connection.execute(
                "SELECT COUNT(*) FROM feedback WHERE review_status='pending'"
            ).fetchone()[0]
            approved = connection.execute(
                "SELECT COUNT(*) FROM feedback WHERE review_status='approved'"
            ).fetchone()[0]
            failed = connection.execute(
                "SELECT COUNT(*) FROM inspections WHERE state='detection_failed'"
            ).fetchone()[0]
            feedback_total = connection.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        return {
            "tenants_active": int(tenants_active),
            "inspections": int(inspections),
            "confirmed": int(confirmed),
            "feedback_pending": int(pending),
            "feedback_approved": int(approved),
            "detection_failed": int(failed),
            "feedback_total": int(feedback_total),
        }

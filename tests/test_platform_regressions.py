from __future__ import annotations

import copy
import json
import sqlite3
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
from urllib.request import Request, urlopen

from platform_app.database import PlatformDatabase, utc_now
from platform_app.http_server import build_server
from platform_app.service import PlatformError, PlatformService


ROOT = Path(__file__).resolve().parents[1]


def owner() -> dict[str, object]:
    return {"role": "admin", "admin": {"access_level": "owner"}}


class ExportStore:
    def __init__(self, root: Path) -> None:
        self.root = root


class ManualCompletionStore:
    def __init__(self) -> None:
        self.label = {
            "annotation_status": "in_progress",
            "geometry": {},
            "assessment": {},
            "classification": {},
            "custom_metadata": {},
            "annotation": {"revision": 0},
        }

    def get_label(self, _project_id: str, _sample_id: str) -> dict[str, object]:
        return copy.deepcopy(self.label)

    def save_label(
        self,
        _project_id: str,
        _sample_id: str,
        label: dict[str, object],
        expected_revision: int,
    ) -> tuple[dict[str, object], bool]:
        if expected_revision != 0:
            raise AssertionError("unexpected revision")
        saved = copy.deepcopy(label)
        saved["annotation"]["revision"] = 1
        self.label = copy.deepcopy(saved)
        return saved, True


class PlatformRegressionTests(unittest.TestCase):
    def test_admin_page_removed_auto_training_and_history_export(self) -> None:
        html = (ROOT / "web" / "admin.html").read_text(encoding="utf-8")
        javascript = (ROOT / "web" / "admin.js").read_text(encoding="utf-8")
        self.assertNotIn("data-admin-view=\"training\"", html)
        self.assertNotIn("trainingAdminView", html)
        self.assertNotIn("includeHistory", html)
        self.assertNotIn("history_limit=", javascript)
        self.assertNotIn('C.api("/admin/training")', javascript)
        self.assertNotIn('C.api("/admin/training/settings"', javascript)
        self.assertNotIn('C.api("/admin/training/start"', javascript)
        self.assertNotIn('C.api("/admin/training/rollback"', javascript)
        self.assertIn('C.api("/admin/training-export"', javascript)

    def test_feedback_selection_does_not_rebuild_scrolled_list(self) -> None:
        javascript = (ROOT / "web" / "admin.js").read_text(encoding="utf-8")
        select_block = javascript.split("function selectFeedback(item)", 1)[1].split(
            "async function deleteSelectedFeedback", 1
        )[0]
        self.assertNotIn("renderList()", select_block)
        self.assertIn("updateFeedbackListSelection()", select_block)

    def test_reviewer_can_complete_an_incomplete_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ManualCompletionStore()
            service = PlatformService(store, Path(directory) / "workspace")
            source_size = {"width": 640, "height": 900}
            manual_outer = [[8, 7], [631, 8], [632, 891], [7, 892]]
            manual_inner = {"left": 25.0, "right": 604.0, "top": 21.1, "bottom": 854.0}
            inspection = {
                "id": "ins_incomplete",
                "tenant_id": "ten_incomplete",
                "project_id": "prj_incomplete",
                "sample_id": "img_incomplete",
                "model_version": "regression-incomplete",
                # Reproduces the reported state: outer geometry exists while the
                # original inner prediction is absent.
                "prediction": {
                    "source_size": source_size,
                    "outer_corners": copy.deepcopy(manual_outer),
                },
            }

            with self.assertRaises(PlatformError) as confirmation_required:
                service._save_accepted_label(
                    inspection,
                    labeler="enterprise",
                    reviewer="reviewer",
                    corrected_inner=copy.deepcopy(manual_inner),
                    corrected_outer=copy.deepcopy(manual_outer),
                    approve_training=True,
                    issue_tags=["inner_frame_wrong"],
                    notes="manual completion regression",
                    manual_completion_confirmed=False,
                )
            self.assertEqual(
                confirmation_required.exception.code,
                "MANUAL_COMPLETION_CONFIRMATION_REQUIRED",
            )

            label, _ = service._save_accepted_label(
                inspection,
                labeler="enterprise",
                reviewer="reviewer",
                corrected_inner=copy.deepcopy(manual_inner),
                corrected_outer=copy.deepcopy(manual_outer),
                approve_training=True,
                issue_tags=["inner_frame_wrong"],
                notes="manual completion regression",
                manual_completion_confirmed=True,
            )
            self.assertEqual(label["geometry"]["outer_corners"], manual_outer)
            self.assertEqual(label["annotation"]["inner_source"], "manual")
            self.assertEqual(
                label["custom_metadata"]["platform"]["manual_completion"],
                {"outer": False, "inner": True},
            )

        javascript = (ROOT / "web" / "admin.js").read_text(encoding="utf-8")
        self.assertIn("manual_completion_confirmed: missingOuter || missingInner", javascript)
        self.assertIn("本次画面上的外框四角和内框四线将作为全人工标注写入训练池", javascript)

    def test_delete_detaches_foreign_key_references(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = PlatformDatabase(Path(directory) / "platform.sqlite3")
            now = utc_now()
            tenant = database.create_tenant("Regression", "prj_regression", "token-value")
            tenant_id = str(tenant["id"])
            with database.connect() as connection:
                connection.execute(
                    "INSERT INTO inspections(id,tenant_id,project_id,sample_id,filename,state,model_version,prediction_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    ("ins_regression", tenant_id, "prj_regression", "img_regression", "card.png", "feedback_pending", "test", "{}", now, now),
                )
                connection.execute(
                    "INSERT INTO feedback(id,inspection_id,issue_tags_json,notes,review_status,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                    ("fbk_regression", "ins_regression", "[]", "", "pending", now, now),
                )
                connection.execute(
                    "INSERT INTO batch_jobs(id,tenant_id,state,expected_count,total_bytes,created_at,updated_at,last_error) VALUES(?,?,?,?,?,?,?,?)",
                    ("bat_regression", tenant_id, "completed", 1, 1, now, now, ""),
                )
                connection.execute(
                    "INSERT INTO batch_items(id,batch_id,position,client_key,filename,content_type,expected_bytes,state,inspection_id,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    ("bti_regression", "bat_regression", 0, "key", "card.png", "image/png", 1, "completed", "ins_regression", now, now),
                )
                connection.execute(
                    "INSERT INTO reference_registration_jobs(id,tenant_id,project_id,state,capture_filename,reference_filename,inspection_id,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                    ("reg_regression", tenant_id, "prj_regression", "completed", "card.png", "ref.png", "ins_regression", now, now),
                )
            deleted = database.delete_feedback_inspection("fbk_regression")
            self.assertEqual(deleted["inspection_id"], "ins_regression")
            with database.connect() as connection:
                self.assertIsNone(connection.execute("SELECT inspection_id FROM batch_items WHERE id='bti_regression'").fetchone()[0])
                self.assertIsNone(connection.execute("SELECT inspection_id FROM reference_registration_jobs WHERE id='reg_regression'").fetchone()[0])
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM inspections WHERE id='ins_regression'").fetchone()[0], 0)

    def test_tenant_delete_removes_reference_jobs_before_inspections(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = PlatformDatabase(Path(directory) / "platform.sqlite3")
            now = utc_now()
            tenant = database.create_tenant("Tenant Delete", "prj_delete", "delete-token")
            tenant_id = str(tenant["id"])
            with database.connect() as connection:
                connection.execute(
                    "INSERT INTO inspections(id,tenant_id,project_id,sample_id,filename,state,model_version,prediction_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    ("ins_tenant_delete", tenant_id, "prj_delete", "img_delete", "card.png", "completed", "test", "{}", now, now),
                )
                connection.execute(
                    "INSERT INTO reference_registration_jobs(id,tenant_id,project_id,state,capture_filename,reference_filename,inspection_id,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                    ("reg_tenant_delete", tenant_id, "prj_delete", "completed", "card.png", "ref.png", "ins_tenant_delete", now, now),
                )
            counts = database.delete_tenant(tenant_id)
            self.assertEqual(counts["inspections"], 1)
            with database.connect() as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM reference_registration_jobs WHERE id='reg_tenant_delete'").fetchone()[0], 0)
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM tenants WHERE id=?", (tenant_id,)).fetchone()[0], 0)

    def test_training_pool_query_is_not_limited_to_one_thousand_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = PlatformDatabase(Path(directory) / "platform.sqlite3")
            now = utc_now()
            tenant = database.create_tenant("Large Pool", "prj_large_pool", "large-pool-token")
            tenant_id = str(tenant["id"])
            count = 1005
            with database.connect() as connection:
                connection.executemany(
                    "INSERT INTO inspections(id,tenant_id,project_id,sample_id,filename,state,model_version,prediction_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    [
                        (
                            f"ins_{index}", tenant_id, "prj_large_pool", f"img_{index}",
                            f"card_{index}.png", "feedback_approved", "test", "{}", now, now,
                        )
                        for index in range(count)
                    ],
                )
                connection.executemany(
                    "INSERT INTO feedback(id,inspection_id,issue_tags_json,notes,review_status,created_at,updated_at,exported_feedback_id) VALUES(?,?,?,?,?,?,?,?)",
                    [
                        (f"fbk_{index}", f"ins_{index}", "[]", "", "approved", now, now, f"export_{index}")
                        for index in range(count)
                    ],
                )
            self.assertEqual(len(database.list_training_feedback()), count)

    def test_training_export_contains_pool_only_and_is_file_backed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store_root = root / "studio"
            store = ExportStore(store_root)
            service = PlatformService(store, root / "workspace")
            feedback_id = "prj_regression_img_regression_r1"
            source_root = store_root / "ml_feedback" / "prj_regression"
            (source_root / "annotations").mkdir(parents=True)
            (source_root / "original_images").mkdir(parents=True)
            (source_root / "rectified_images").mkdir(parents=True)
            original = source_root / "original_images" / f"{feedback_id}.png"
            rectified = source_root / "rectified_images" / f"{feedback_id}.png"
            original.write_bytes(b"original")
            rectified.write_bytes(b"rectified")
            (source_root / "annotations" / f"{feedback_id}.json").write_text(
                json.dumps(
                    {
                        "image": {"path": f"original_images/{feedback_id}.png"},
                        "rectification": {"image_path": f"rectified_images/{feedback_id}.png"},
                    }
                ),
                encoding="utf-8",
            )
            approved = {
                "id": "fbk_regression",
                "project_id": "prj_regression",
                "exported_feedback_id": feedback_id,
            }
            conversion = {"converted": {"outer_pose": 1, "outer_seg": 1, "inner_seg": 1, "inner_refiner": 1}}

            def fake_convert(_feedback_root: Path, output: Path, split: str = "train") -> dict[str, object]:
                self.assertEqual(split, "train")
                for dataset in ("outer_pose", "outer_seg", "inner_seg"):
                    target = output / dataset
                    target.mkdir(parents=True, exist_ok=True)
                    (target / "data.yaml").write_text("path: absolute\n", encoding="utf-8")
                (output / "inner_refiner_manifest.csv").write_text("id,source\none,feedback\n", encoding="utf-8")
                return conversion

            feedback_dataset = ModuleType("feedback_dataset")
            feedback_dataset.convert_feedback_to_training = fake_convert

            with patch.object(service.database, "list_training_feedback", return_value=[approved]), patch.dict(
                "sys.modules", {"feedback_dataset": feedback_dataset}
            ):
                archive_path, filename = service.export_training_bundle(owner())

            self.assertIsInstance(archive_path, Path)
            self.assertTrue(archive_path.is_file())
            self.assertIn("training_pool", filename)
            try:
                with zipfile.ZipFile(archive_path) as archive:
                    names = archive.namelist()
                    self.assertFalse(any("history_" in name for name in names))
                    manifest = json.loads(archive.read("export_manifest.json"))
                    self.assertEqual(manifest["data_scope"], "approved_training_pool_only")
                    self.assertFalse(manifest["historical_data_included"])
            finally:
                archive_path.unlink(missing_ok=True)

    def test_http_download_streams_file_and_removes_temporary_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "export.zip"
            expected = b"streamed-export" * 100_000
            archive_path.write_bytes(expected)

            class DownloadService:
                @staticmethod
                def authenticate(_token: str) -> dict[str, object]:
                    return owner()

                @staticmethod
                def authenticate_session(_token: str) -> dict[str, object]:
                    return owner()

                @staticmethod
                def export_training_bundle(_principal: dict[str, object]) -> tuple[Path, str]:
                    return archive_path, "CardScope_training_pool_test.zip"

            server = build_server(DownloadService(), "127.0.0.1", 0, ROOT / "web", 1024, 1024)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                port = int(server.server_address[1])
                request = Request(
                    f"http://127.0.0.1:{port}/api/platform/v1/admin/training-export",
                    headers={"X-Platform-Token": "adm_test"},
                )
                with urlopen(request, timeout=10) as response:
                    self.assertEqual(response.read(), expected)
                    self.assertIn("CardScope_training_pool_test.zip", response.headers["Content-Disposition"])
                for _ in range(50):
                    if not archive_path.exists():
                        break
                    time.sleep(0.01)
                self.assertFalse(archive_path.exists())
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_removed_auto_training_routes_return_not_found(self) -> None:
        class RouteService:
            @staticmethod
            def authenticate(_token: str) -> dict[str, object]:
                return owner()

            @staticmethod
            def authenticate_session(_token: str) -> dict[str, object]:
                return owner()

        server = build_server(RouteService(), "127.0.0.1", 0, ROOT / "web", 1024, 1024)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        port = server.server_address[1]
        try:
            for method, route, body in (
                ("GET", "/api/platform/v1/admin/training", None),
                ("POST", "/api/platform/v1/admin/training/settings", b"{}"),
                ("POST", "/api/platform/v1/admin/training/start", b"{}"),
                ("POST", "/api/platform/v1/admin/training/rollback", b"{}"),
            ):
                request = Request(
                    f"http://127.0.0.1:{port}{route}",
                    method=method,
                    data=body,
                    headers={"X-Platform-Token": "owner-token", "Content-Type": "application/json"},
                )
                with self.assertRaises(Exception) as caught:
                    urlopen(request, timeout=5)
                self.assertEqual(getattr(caught.exception, "code", None), 404)
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)


if __name__ == "__main__":
    unittest.main()

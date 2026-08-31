from __future__ import annotations

import json
from pathlib import Path

from app.services import storage


def test_save_project_retries_transient_windows_replace_lock(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(storage, "PROJECTS_ROOT", tmp_path)
    real_replace = storage.os.replace
    attempts = 0

    def flaky_replace(source, destination):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError(5, "transient Windows file lock")
        return real_replace(source, destination)

    monkeypatch.setattr(storage.os, "replace", flaky_replace)
    project = {"id": "save-retry", "name": "retry"}
    storage.save_project(project)

    saved = json.loads((tmp_path / "save-retry" / "project.json").read_text(encoding="utf-8"))
    assert saved["name"] == "retry"
    assert attempts == 2
    assert not list((tmp_path / "save-retry").glob("*.tmp"))

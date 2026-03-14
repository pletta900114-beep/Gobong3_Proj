from __future__ import annotations

from pathlib import Path

import pytest

from mellow_chat_runtime.infra import database as database_module


@pytest.fixture(autouse=True)
def override_test_database(tmp_path: Path):
    original_url = database_module.DATABASE_URL
    test_db_path = tmp_path / "test_chatbot.db"
    database_module.configure_database(f"sqlite:///{test_db_path}")
    yield
    database_module.configure_database(original_url)

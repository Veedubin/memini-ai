import os
import re

import asyncpg
import pytest
import structlog

from memini_ai.db import Database

TEST_DSN = os.environ.get(
    "MEMINI_TEST_DB_URL", "postgresql://memini:memini@localhost:5555/memini_test"
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No MEMINI_* setting from the developer's shell may reach a test.

    MEMINI_TEST_DB_URL is the one exception: it is how CI points the suite at its own database.
    """
    for name in list(os.environ):
        if name.startswith("MEMINI_") and name != "MEMINI_TEST_DB_URL":
            monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def reset_structlog():
    """configure_logging() binds structlog to the current sys.stderr.

    Under capsys that is a buffer which is closed when the test ends, so a test that configures
    logging would break every later test's log call. Reset after each test.
    """
    yield
    structlog.reset_defaults()


@pytest.fixture(scope="session")
def test_dsn() -> str:
    return TEST_DSN


@pytest.fixture(scope="session")
def bad_password_dsn(test_dsn: str) -> str:
    """The configured DSN with a wrong password: connecting must fail, wherever the DB lives."""
    return re.sub(r"://([^:]+):[^@]*@", r"://\1:wrong-password@", test_dsn)


@pytest.fixture(scope="session")
def missing_db_dsn(test_dsn: str) -> str:
    """The configured DSN pointing at a database that does not exist."""
    base, _, _ = test_dsn.rpartition("/")
    return f"{base}/memini_does_not_exist"


@pytest.fixture
async def db(test_dsn: str):
    database = Database(test_dsn)
    await database.connect()
    yield database
    await database.execute("TRUNCATE memories, chains CASCADE")
    await database.close()


@pytest.fixture
async def raw_conn(test_dsn: str):
    conn = await asyncpg.connect(test_dsn)
    yield conn
    await conn.close()

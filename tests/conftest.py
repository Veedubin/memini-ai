import os

import asyncpg
import pytest

from memini_ai.db import Database

TEST_DSN = os.environ.get(
    "MEMINI_TEST_DB_URL", "postgresql://memini:memini@localhost:5555/memini_test"
)


@pytest.fixture(scope="session")
def test_dsn() -> str:
    return TEST_DSN


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

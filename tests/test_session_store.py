"""SessionStore: CRUD + claim/release crash recovery."""

import time

import pytest

from opencode_core.database import Database
from opencode_core.session import SessionStore, system_prompt
from opencode_schema.model import ModelRef
from opencode_schema.session import UserMessage


@pytest.fixture
def store(tmp_path):
    db = Database(tmp_path / "test.db")
    yield SessionStore(db, stale_claim_ms=60_000)
    db.close()


def test_create_and_get(store):
    sid = store.create(title="hello", agent="build", model="openai/gpt-4o")
    record = store.get(sid)
    assert record is not None
    assert record.title == "hello"
    assert record.agent == "build"
    assert record.model_str == "openai/gpt-4o"


def test_get_missing_returns_none(store):
    assert store.get("ses_nope") is None


def test_add_and_read_messages(store):
    sid = store.create(title="t")
    store.add_message(sid, UserMessage(text="hi"))
    messages = store.messages(sid)
    assert len(messages) == 1
    assert messages[0].type == "user"
    assert messages[0].text == "hi"


def test_list_newest_first(store):
    a = store.create(title="old")
    time.sleep(0.01)
    b = store.create(title="new")
    summaries = store.list()
    assert summaries[0].id == b
    assert summaries[1].id == a


def test_claim_release_cycle(store):
    sid = store.create(title="t")
    assert store.claim(sid) is True
    # A live claim is not reclaimable.
    assert store.claim(sid) is False
    store.release(sid)
    assert store.claim(sid) is True


def test_stale_claim_is_reclaimable(store):
    store.stale_claim_ms = 1000
    sid = store.create(title="t")
    assert store.claim(sid) is True
    # Simulate the previous owner crashing: age the claim beyond staleness.
    store.db.execute("UPDATE sessions SET time_suspended = ? WHERE id = ?", (time.time() - 5, sid))
    store.db.commit()
    assert store.claim(sid) is True  # reclaimed


def test_claim_missing_session_fails(store):
    assert store.claim("ses_nope") is False


def test_resume_attempts_increment(store):
    sid = store.create(title="t")
    store.claim(sid)
    store.release(sid)
    store.claim(sid)
    store.release(sid)
    row = store.db.execute("SELECT resume_attempts FROM sessions WHERE id = ?", (sid,)).fetchone()
    assert row["resume_attempts"] == 2


def test_update_metadata(store):
    sid = store.create(title="t")
    store.update_metadata(sid, title="renamed", model=ModelRef.parse("anthropic/claude-sonnet-4"))
    record = store.get(sid)
    assert record.title == "renamed"
    assert record.model_str == "anthropic/claude-sonnet-4"


def test_system_prompt_present():
    text = system_prompt()
    assert "coding assistant" in text

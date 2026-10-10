"""Shared work queue: exclusive claims, takeover of dead claims, walltime gate."""

import os
import time

import pytest

from flexcompress.workqueue import ClaimQueue, DeadlineGate, parse_slurm_duration


def _age(path, seconds):
    t = time.time() - seconds
    os.utime(path, (t, t))


def test_claims_are_exclusive_and_released(tmp_path):
    a, b = ClaimQueue(tmp_path), ClaimQueue(tmp_path)
    assert a.try_claim("x") and a.try_claim("x")  # idempotent for the holder
    assert not b.try_claim("x")
    assert b.try_claim("y")
    assert a.live_keys() == {"x", "y"}
    a.release("x")
    assert b.try_claim("x")
    a.release("x")  # not a's claim any more: must not remove b's
    assert (tmp_path / "claims" / "x.claim").is_file() and not a.try_claim("x")


def test_stale_claim_is_taken_over_once(tmp_path):
    owner = ClaimQueue(tmp_path)
    assert owner.try_claim("x")
    _age(tmp_path / "claims" / "x.claim", 60)
    takers = [ClaimQueue(tmp_path, stale_s=30) for _ in range(3)]
    wins = [q.try_claim("x") for q in takers]
    assert wins == [True, False, False]  # the new claim is fresh, so later takers back off
    owner.release("x")  # the dead owner waking up must not drop the new claim
    assert "x" in takers[0].live_keys()


def test_late_taker_never_removes_a_fresh_takeover(tmp_path):
    """Two workers both judged a claim dead; the slower one must not delete the winner's new claim."""

    owner = ClaimQueue(tmp_path)
    assert owner.try_claim("x")
    path = tmp_path / "claims" / "x.claim"
    _age(path, 60)
    fast, slow = ClaimQueue(tmp_path, stale_s=30), ClaimQueue(tmp_path, stale_s=30)
    assert fast.try_claim("x")
    assert not slow._take_over("x", path)  # slow had seen the dead claim before fast replaced it
    assert path.is_file() and fast.held() == {"x"} and slow.held() == set()
    (tmp_path / "claims" / "x.claim.takeover").write_text("")  # a taker died holding the lock
    _age(tmp_path / "claims" / "x.claim.takeover", 60)
    _age(path, 60)
    assert not slow.try_claim("x")  # removes the dead lock, backs off this time
    assert slow.try_claim("x")


def test_claims_of_a_finished_array_are_void(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "1111")
    old = ClaimQueue(tmp_path)
    assert old.try_claim("x")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "2222")
    new = ClaimQueue(tmp_path)
    assert not new.try_claim("x")  # fresh heartbeat, owner may be alive
    new.mark_dead("1111")  # the finalize job records that array 1111 ended
    assert old.live_keys() == set() and new.try_claim("x")


def test_heartbeat_keeps_claims_fresh(tmp_path):
    q = ClaimQueue(tmp_path, heartbeat_s=0.05).start()
    try:
        assert q.try_claim("x")
        _age(tmp_path / "claims" / "x.claim", 120)
        time.sleep(0.3)
        assert not ClaimQueue(tmp_path, stale_s=60).try_claim("x")
    finally:
        q.close()
    assert not (tmp_path / "claims" / "x.claim").exists()  # close() releases


@pytest.mark.parametrize("text,seconds", [("1-02:03:04", 93784.0), ("23:59:00", 86340.0), ("05:30", 330.0),
                                          ("UNLIMITED", None), ("", None), ("INVALID", None)])
def test_parse_slurm_duration(text, seconds):
    assert parse_slurm_duration(text) == seconds


def test_deadline_gate_learns_runtime():
    assert DeadlineGate(None).allows(1e12)
    gate = DeadlineGate(time.time() + 3600, margin_s=600, safety=1.5, first_s=1800)
    assert gate.allows(1.0)  # 1 h left > 30 min before anything has finished
    gate.record(cost=100.0, seconds=1000.0)  # 10 s per unit
    assert gate.allows(100.0)  # 1.5 x 1000 s + 600 s < 3600 s
    assert not gate.allows(200.0)  # 1.5 x 2000 s + 600 s > 3600 s
    assert gate.refused == 1
    assert not DeadlineGate(time.time() + 600, first_s=1800).allows(1.0)

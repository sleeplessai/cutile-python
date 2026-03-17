# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import sqlite3
import time

import pytest

from cuda.tile._cache import cache_key, cache_lookup, cache_store, evict_lru


def test_cache_key_equal():
    k1 = cache_key("v1", "sm_90", 3, b"data")
    k2 = cache_key("v1", "sm_90", 3, b"data")
    assert k1 == k2


def test_cache_key_differs():
    base = cache_key("v1", "sm_90", 3, b"data")
    assert cache_key("v2", "sm_90", 3, b"data") != base
    assert cache_key("v1", "sm_80", 3, b"data") != base
    assert cache_key("v1", "sm_90", 2, b"data") != base
    assert cache_key("v1", "sm_90", 3, b"other") != base


@pytest.fixture
def cache_env(tmp_path):
    cache_dir = str(tmp_path / "cache")
    return cache_dir, tmp_path


def test_store_then_lookup(cache_env):
    cache_dir, tmp_path = cache_env
    key = cache_key("v1", "sm_90", 3, b"data")
    content = b"\x7fELF_fake_cubin_data"

    cache_store(cache_dir, key, content)

    result = cache_lookup(cache_dir, key)
    assert result is not None
    assert result == content


def test_lookup_updates_atime(cache_env):
    cache_dir, tmp_path = cache_env
    key = cache_key("v1", "sm_90", 3, b"data")

    cache_store(cache_dir, key, b"data")

    # Manually set old atime in DB
    import os
    db_path = os.path.join(cache_dir, "cache.db")
    old_time = time.time() - 1000
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE cache SET atime = ? WHERE key = ?", (old_time, key))
    conn.commit()
    conn.close()

    cache_lookup(cache_dir, key)

    conn = sqlite3.connect(db_path)
    atime = conn.execute(
        "SELECT atime FROM cache WHERE key = ?", (key,)
    ).fetchone()[0]
    conn.close()
    assert atime > old_time


def test_lookup_miss(cache_env):
    cache_dir, _ = cache_env

    result = cache_lookup(cache_dir, "a" * 64)
    assert result is None


def test_evict_lru(cache_env):
    cache_dir, tmp_path = cache_env
    import os
    db_path = os.path.join(cache_dir, "cache.db")

    # Populate 5 entries (1000 bytes each, 5000 total)
    keys = []
    for i in range(5):
        key = cache_key(str(i), "sm_90", 3, b"data")
        keys.append(key)
        cache_store(cache_dir, key, b"x" * 1000)

    # Set controlled atimes so eviction order is deterministic
    conn = sqlite3.connect(db_path)
    for i, key in enumerate(keys):
        conn.execute(
            "UPDATE cache SET atime = ? WHERE key = ?",
            (float(i), key)
        )
    conn.commit()
    conn.close()

    # Evict to keep 3000 bytes; newest 3 survive (indices 2, 3, 4)
    evict_lru(cache_dir, 3000)

    remaining = [k for k in keys
                 if cache_lookup(cache_dir, k) is not None]
    assert remaining == keys[2:]

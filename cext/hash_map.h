/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "memory.h"
#include "hash.h"

#include <new>
#include <utility>


template <typename A, typename B>
struct CompareKey {};

template <typename K>
struct CompareKey<K, K> {
    static bool equals(const K& a, const K& b) {
        return a == b;
    }
};


template <typename K, typename V>
class HashMap {
    static constexpr uint64_t kOccupiedBit = 1ull << 63;
    static constexpr size_t kInitialBuckets = 4;

public:
    struct Item {
        const K key;
        V value;

        Item(K&& k, V&& v) : key(std::move(k)), value(std::move(v)) {}
        Item(Item&&) = default;

        Item(const Item&) = delete;
        void operator=(const Item&) = delete;
    };

    HashMap()
      : nbuckets_(kInitialBuckets),
        size_(0),
        hashes_(xcalloc<uint64_t>(kInitialBuckets)),
        items_(xcalloc<Item>(kInitialBuckets))
    {}

    HashMap(const HashMap&) = delete;
    void operator= (const HashMap&) = delete;

    ~HashMap() {
        for (size_t i = 0; i < nbuckets_; ++i) {
            if (hashes_[i] & kOccupiedBit)
                items_[i].~Item();
        }
        mem_free(hashes_);
        mem_free(items_);
    }

    template <typename Q>
    Item* find(Q&& key) {
        uint64_t needle = compute_hash(key) | kOccupiedBit;
        auto [found, pos] = lookup(key, needle);
        return found ? &items_[pos] : nullptr;
    }

    Item* insert(K key, V value) {
        uint64_t needle = compute_hash(key) | kOccupiedBit;
        auto [found, pos] = lookup(key, needle);
        if (found) {
            return &items_[pos];
        } else {
            if (size_ >= nbuckets_ / 2) {
                rehash();
                auto [found_rehashed, pos_rehashed] = lookup(key, needle);
                CHECK(!found_rehashed);
                pos = pos_rehashed;
            }
            ++size_;
            hashes_[pos] = needle;
            return new (&items_[pos]) Item(std::move(key), std::move(value));
        }
    }

private:

    size_t nbuckets_;  // power of two
    size_t size_;
    uint64_t* hashes_;
    Item* items_;

    template <typename A, typename B>
    static bool key_equals(const A& a, const B& b) {
        return CompareKey<A, B>::equals(a, b);
    }

    template <typename Q>
    std::pair<bool, size_t> lookup(Q&& key, uint64_t needle) {
        size_t pos = needle & (nbuckets_ - 1);
        while (true) {
            uint64_t h = hashes_[pos];
            if (!(h & kOccupiedBit)) return {false, pos};
            if (h == needle && key_equals(key, items_[pos].key)) return {true, pos};
            pos = (pos + 1) & (nbuckets_ - 1);
        }
    }

    void rehash() {
        size_t old_nbuckets = nbuckets_;
        CHECK(old_nbuckets < SIZE_MAX / 2);
        size_t new_nbuckets = old_nbuckets * 2;

        uint64_t* new_hashes = xcalloc<uint64_t>(new_nbuckets);
        Item* new_items = xcalloc<Item>(new_nbuckets);

        uint64_t* old_hashes = hashes_;
        Item* old_items = items_;
        for (size_t i = 0; i < old_nbuckets; ++i) {
            size_t h = old_hashes[i];
            if (h & kOccupiedBit) {
                size_t new_pos = h & (new_nbuckets - 1);
                while (new_hashes[new_pos] & kOccupiedBit)
                    new_pos = (new_pos + 1) & (new_nbuckets - 1);
                new (new_items + new_pos) Item(std::move(old_items[i]));
                old_items[i].~Item();
                new_hashes[new_pos] = h;
            }
        }

        mem_free(old_hashes);
        mem_free(old_items);
        hashes_ = new_hashes;
        items_ = new_items;
        nbuckets_ = new_nbuckets;
    }

    template <typename Q>
    static uint64_t compute_hash(const Q& key) {
        Hasher hasher;
        Hash<Q>::hash(key, hasher);
        return hasher.get();
    }
};


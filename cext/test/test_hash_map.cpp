// SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0


#include "../hash_map.h"
#include "../check.h"


int main() {
    HashMap<int, int> hm;
    CHECK(!hm.find(1));

    hm.insert(0, 0);
    CHECK(hm.find(0));
    CHECK(hm.find(0)->value == 0);

    // Insert doesn't overwrite existing values
    hm.insert(0, 20);
    CHECK(hm.find(0)->value == 0);

    for (int i = 1; i < 1000; ++i) {
        hm.insert(i * 16, i * 10);
        for (int j = 0; j <= i; ++j) {
            auto* item = hm.find(j * 16);
            CHECK(item);
            CHECK(item->value == j * 10);
        }
    }

    return 0;
}


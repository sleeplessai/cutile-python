/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "py.h"
#include <cuda.h>

#define FOREACH_CUDA_FUNCTION_TO_LOAD(X) \
    X(cuInit, 2000) \
    X(cuLibraryLoadData, 12000) \
    X(cuLibraryUnload, 12000) \
    X(cuLibraryGetKernel, 12000) \
    X(cuGetErrorString, 6000) \
    X(cuLaunchKernel, 7000) \
    X(cuPointerGetAttribute, 4000) \
    X(cuCtxPushCurrent, 4000) \
    X(cuCtxPopCurrent, 4000) \
    X(cuCtxGetCurrent, 4000) \
    X(cuCtxSetCurrent, 4000) \
    X(cuCtxGetDevice, 2000) \
    X(cuCtxGetId, 12000) \
    X(cuDeviceGet, 2000) \
    X(cuDeviceGetAttribute, 2000) \
    X(cuDevicePrimaryCtxRetain, 7000) \
    X(cuDriverGetVersion, 2020) \
    X(cuEventCreate, 2000) \
    X(cuEventDestroy, 2000) \
    X(cuEventQuery, 2000) \
    X(cuEventRecord, 2000) \
    X(cuMemAlloc, 3020) \
    X(cuMemAllocHost, 3020) \
    X(cuMemFree, 3020) \
    X(cuMemFreeHost, 2000) \
    X(cuMemcpyHtoDAsync, 3020) \
    X(cuStreamCreate, 2000) \
    X(cuStreamGetCtx, 9020) \
    X(cuStreamGetId, 12000) \
    X(cuStreamIsCapturing, 10000) \
    X(cuStreamSynchronize, 7000) \
    X(cuStreamWaitEvent, 7000)


#define DECLARE_CUDA_FUNC_EXTERN(name, _cuda_version) \
    decltype(::name)* name;

struct DriverApi {
    FOREACH_CUDA_FUNCTION_TO_LOAD(DECLARE_CUDA_FUNC_EXTERN)
};

Result<const DriverApi*> get_driver_api();

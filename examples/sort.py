#!/usr/bin/env python

# Copyright 2022 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse

import numpy as np
from benchmark import run_benchmark
from legate.timing import time

import cunumeric as num


def check_sorted(a, a_sorted, axis=-1):
    a_np = a.__array__()
    a_np_sorted = np.sort(a_np, axis)
    print("Checking result...")
    if num.allclose(a_np_sorted, a_sorted):
        print("PASS!")
    else:
        print("FAIL!")
        print("NUMPY    : " + str(a_np_sorted))
        print("CUNUMERIC: " + str(a_sorted))
        assert False


def run_sort(N, shape, axis, datatype, lower, upper, perform_check, timing):

    num.random.seed(42)
    newtype = np.dtype(datatype).type
    if shape is not None:
        shape = tuple(shape)
    else:
        shape = (N,)

    if np.issubdtype(newtype, np.integer):
        if lower is None:
            lower = np.iinfo(newtype).min
        if upper is None:
            upper = np.iinfo(newtype).max
        a = num.random.randint(low=lower, high=upper, size=N).astype(newtype)
        a = a.reshape(shape)
    elif np.issubdtype(newtype, np.floating):
        a = num.random.random(shape).astype(newtype)
    elif np.issubdtype(newtype, np.complexfloating):
        a = num.array(
            num.random.random(shape) + num.random.random(shape) * 1j
        ).astype(newtype)
    else:
        print("UNKNOWN type " + str(newtype))
        assert False

    start = time()
    a_sorted = num.sort(a, axis)
    stop = time()

    if perform_check:
        check_sorted(a, a_sorted, axis)
    else:
        # do we need to synchronize?
        assert True
    total = (stop - start) * 1e-3
    if timing:
        print("Elapsed Time: " + str(total) + " ms")
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--check",
        dest="check",
        action="store_true",
        help="check the result of the solve",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=1000000,
        dest="N",
        help="number of elements in one dimension",
    )
    parser.add_argument(
        "-t",
        "--time",
        dest="timing",
        action="store_true",
        help="perform timing",
    )
    parser.add_argument(
        "-s",
        "--shape",
        type=int,
        nargs="+",
        default=None,
        dest="shape",
        help="array reshape (default 'None')",
    )
    parser.add_argument(
        "-d",
        "--datatype",
        type=str,
        default="uint32",
        dest="datatype",
        help="data type (default np.int32)",
    )
    parser.add_argument(
        "-l",
        "--lower",
        type=int,
        default=None,
        dest="lower",
        help="lower bound for integer based arrays (inclusive)",
    )
    parser.add_argument(
        "-u",
        "--upper",
        type=int,
        default=None,
        dest="upper",
        help="upper bound for integer based arrays (exclusive)",
    )
    parser.add_argument(
        "-a",
        "--axis",
        type=int,
        default=-1,
        dest="axis",
        help="sort axis (default -1)",
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        type=int,
        default=1,
        dest="benchmark",
        help="number of times to benchmark this application (default 1 - "
        "normal execution)",
    )

    args = parser.parse_args()
    run_benchmark(
        run_sort,
        args.benchmark,
        "Sort",
        (
            args.N,
            args.shape,
            args.axis,
            args.datatype,
            args.lower,
            args.upper,
            args.check,
            args.timing,
        ),
    )

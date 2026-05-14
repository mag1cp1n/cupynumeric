from __future__ import annotations

import argparse
import importlib
import math
import sys
import warnings

from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_DIR = _REPO_ROOT / "examples"
_MICROBENCHMARKS_DIR = _EXAMPLES_DIR / "microbenchmarks"


def _ensure_example_paths() -> None:
    for path in (str(_MICROBENCHMARKS_DIR), str(_EXAMPLES_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)


@lru_cache
def _module(name: str):
    _ensure_example_paths()
    return importlib.import_module(name)


def _sizing():
    return _module("_benchmark.sizing")


def _build_size_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    _sizing().add_size_request_parser_group(parser)
    return parser


def _next_size_with_larger_estimate(estimate, size: int) -> int:
    current = estimate(size)
    limit = size + max(1024, 2 * math.isqrt(max(size, 1)) + 16)
    for candidate in range(size + 1, limit + 1):
        if estimate(candidate) > current:
            return candidate
    raise AssertionError("failed to find a larger estimated working set")


def _assert_target_resolution(resolve, estimate, target_bytes: int) -> None:
    size = resolve(target_bytes)
    assert estimate(size) <= target_bytes
    larger_size = _next_size_with_larger_estimate(estimate, size)
    assert estimate(larger_size) > target_bytes


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1B", 1),
        ("2KiB", 2 << 10),
        ("3MiB", 3 << 20),
        ("4GiB", 4 << 30),
        ("5TiB", 5 << 40),
    ],
)
def test_parse_memory_size_valid(value: str, expected: int) -> None:
    assert _sizing().parse_memory_size(value) == expected


@pytest.mark.parametrize("value", ["0B", "1KB", "1Mi", "GiB", "-1MiB"])
def test_parse_memory_size_invalid(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _sizing().parse_memory_size(value)


def test_size_request_defaults_to_legacy_exact_size() -> None:
    args = _build_size_parser().parse_args([])
    benchmark = _sizing()
    request = benchmark.SizeRequest.from_namespace(args)

    assert request.exact_size == [benchmark.DEFAULT_PROBLEM_SIZE]
    assert request.memory_target_bytes is None
    assert request.config_lines() == [
        "Sizing: exact (--size)",
        f"Suite-defined size: {benchmark.DEFAULT_PROBLEM_SIZE:,}",
    ]


def test_size_request_help_mentions_legacy_default() -> None:
    help_text = _build_size_parser().format_help()

    assert "default: 10,000,000" in help_text
    assert "default: None" not in help_text


def test_size_request_parses_memory_target_mode() -> None:
    args = _build_size_parser().parse_args(["--memory-size", "2GiB"])
    request = _sizing().SizeRequest.from_namespace(args)

    assert request.exact_size is None
    assert request.memory_target_bytes == [2 << 30]


def test_size_request_parses_capacity_sweeps() -> None:
    args = _build_size_parser().parse_args(["--memory-size", "1KiB", "2KiB"])
    request = _sizing().SizeRequest.from_namespace(args)

    assert request.exact_size is None
    assert request.memory_target_bytes == [1 << 10, 2 << 10]
    assert request.config_lines() == [
        "Sizing: working-set target (--memory-size)",
        "Approximate working-set target (bytes): 1,024, 2,048",
    ]


def test_size_request_parses_work_rescale() -> None:
    args = _build_size_parser().parse_args(
        ["--memory-size", "2GiB", "--rescale-by-work", "1", "2.5"]
    )
    request = _sizing().SizeRequest.from_namespace(args)

    assert request.memory_target_bytes == [2 << 30]
    assert request.rescale_by_work == [1.0, 2.5]
    assert request.config_lines() == [
        "Sizing: working-set target (--memory-size)",
        f"Approximate working-set target (bytes): {2 << 30:,}",
        "Work rescale factors: 1.0, 2.5",
    ]


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_size_request_parser_rejects_invalid_work_rescale(value: str) -> None:
    parser = _build_size_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--rescale-by-work", value])


def test_size_request_rejects_programmatic_invalid_work_rescale() -> None:
    args = SimpleNamespace(rescale_by_work=[0.0])
    with pytest.raises(RuntimeError, match="positive"):
        _sizing().SizeRequest.from_namespace(args)


def test_size_request_rejects_programmatic_empty_work_rescale() -> None:
    with pytest.raises(RuntimeError, match="at least one"):
        _sizing().SizeRequest(exact_size=[1], rescale_by_work=[])


def test_size_request_requires_mode_when_default_is_disabled() -> None:
    args = _build_size_parser().parse_args([])
    with pytest.raises(RuntimeError, match="size request must specify"):
        _sizing().SizeRequest.from_namespace(args, default_exact_size=None)


def test_size_request_parser_rejects_conflicting_modes() -> None:
    parser = _build_size_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--size", "32", "--memory-size", "1MiB"])


@pytest.mark.parametrize("flag", ["--size", "--memory-size"])
def test_size_request_parser_rejects_missing_size_values(flag: str) -> None:
    parser = _build_size_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([flag])


def _make_recording_suite(
    *, forbid_resolution: bool = False
) -> SimpleNamespace:
    calls: list[tuple[str, tuple[object, ...]]] = []
    call_records: list[object] = []
    info_names: list[str] = []
    resolutions: list[object] = []

    def print_size_resolution(resolution: object) -> None:
        if forbid_resolution:
            raise AssertionError(
                "exact-size path should not print a resolution"
            )
        resolutions.append(resolution)

    def run_timed(func, *args, **kwargs) -> None:
        del kwargs
        calls.append((func.__name__, args))

    def run_timed_with_info(info, func, *args, **kwargs) -> None:
        del kwargs
        info_names.append(info.name)
        calls.append((func.__name__, args))

    def run_timed_with_generator(info, func, gen, **kwargs) -> None:
        del kwargs
        if info is not None:
            info_names.append(info.name)
        for args in gen:
            calls.append((func.__name__, args))

    def run_timed_calls(planned_calls, **kwargs) -> None:
        del kwargs
        utilities = _module("_benchmark.microbenchmark_utilities")
        for func, group in utilities._group_microbenchmark_calls(
            planned_calls
        ):
            call_records.extend(group)
            info_names.append(group[0].name)
            for call in group:
                calls.append((func.__name__, call.args))

    return SimpleNamespace(
        np=np,
        timer=object(),
        runs=1,
        warmup=0,
        calls=calls,
        call_records=call_records,
        info_names=info_names,
        resolutions=resolutions,
        print_size_resolution=print_size_resolution,
        run_timed=run_timed,
        run_timed_with_info=run_timed_with_info,
        run_timed_with_generator=run_timed_with_generator,
        run_timed_calls=run_timed_calls,
        _config=SimpleNamespace(package="numpy"),
    )


class _TransposeStub:
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def T(self) -> str:
        return f"{self._name}.T"


@pytest.mark.parametrize(
    (
        "module_name",
        "runner_kwargs",
        "expected_name",
        "size_index",
        "num_resolutions",
    ),
    [
        ("general_astype_bench", {}, "astype", 2, 1),
        ("general_random_bench", {}, "randint", 1, 1),
        ("general_nanred_bench", {}, "nan_red", 3, 1),
        ("general_scalared_bench", {}, "scalar_red", 3, 1),
        ("ufunc_bench", {"perform_check": False}, "unary_exp", 1, 1),
        ("general_indexing_bench", {}, "boolean_set_array", 1, 1),
        # fast_advanced_indexing_bench has two resolution groups: _BPE_9 (putmask)
        # and _BPE_24 (boolean get tests), so it emits two resolutions.
        ("fast_advanced_indexing_bench", {}, "putmask_scalar", 1, 2),
    ],
)
def test_linear_suites_resolve_memory_target_in_run_benchmarks(
    module_name: str,
    runner_kwargs: dict[str, object],
    expected_name: str,
    size_index: int,
    num_resolutions: int,
) -> None:
    module = _module(module_name)
    sizing = _sizing()
    suite = _make_recording_suite()
    request = sizing.SizeRequest(memory_target_bytes=[100])

    module.run_benchmarks(suite, request, **runner_kwargs)

    assert len(suite.resolutions) == num_resolutions
    assert len(suite.resolutions[0]) == 1
    resolution = suite.resolutions[0][0]
    assert resolution.requested_memory_target_bytes == 100
    assert resolution.estimated_working_set_bytes <= 100
    assert suite.calls
    call_name, call_args = suite.calls[0]
    assert call_name == expected_name
    assert call_args[size_index] == [resolution.resolved_size]


def test_sort_suite_resolves_memory_target_in_run_benchmarks() -> None:
    sort_bench = _module("sort_bench")
    suite = _make_recording_suite()
    target_bytes = 1 << 20
    request = _sizing().SizeRequest(memory_target_bytes=[target_bytes])

    sort_bench.run_benchmarks(suite, request, variant="all", precision="all")

    assert len(suite.resolutions) == 1
    assert len(suite.resolutions[0]) == 1
    resolution = suite.resolutions[0][0]
    assert resolution.requested_memory_target_bytes == target_bytes
    assert resolution.estimated_working_set_bytes <= target_bytes
    assert suite.calls
    call_name, call_args = suite.calls[0]
    assert call_name == "sort"
    assert call_args[2] == [resolution.resolved_size]

    def estimate(size: int) -> int:
        return sort_bench._estimate_working_set_bytes("all", "all", size)

    larger_size = _next_size_with_larger_estimate(
        estimate, resolution.resolved_size
    )
    assert estimate(larger_size) > target_bytes


def test_axis_sum_suite_resolves_memory_target_in_run_benchmarks() -> None:
    axis_sum = _module("axis_sum_bench")
    suite = _make_recording_suite()
    target_bytes = 1 << 20
    request = _sizing().SizeRequest(memory_target_bytes=[target_bytes])

    axis_sum.run_benchmarks(suite, request, case="all", perform_check=False)

    assert len(suite.resolutions) == 1
    assert len(suite.resolutions[0]) == 1
    resolution = suite.resolutions[0][0]
    assert resolution.requested_memory_target_bytes == target_bytes
    assert resolution.estimated_working_set_bytes <= target_bytes
    assert suite.calls
    call_name, call_args = suite.calls[0]
    assert call_name == "axis_sum"
    assert call_args[6] == resolution.resolved_size

    def estimate(size: int) -> int:
        return axis_sum._estimate_working_set_bytes("all", size)

    larger_size = _next_size_with_larger_estimate(
        estimate, resolution.resolved_size
    )
    assert estimate(larger_size) > target_bytes


def test_batched_fft_suite_resolves_memory_target_in_run_benchmarks() -> None:
    batched_fft = _module("batched_fft_bench")
    suite = _make_recording_suite()
    target_bytes = 8 << 20
    request = _sizing().SizeRequest(memory_target_bytes=[target_bytes])

    batched_fft.run_benchmarks(suite, request)

    assert len(suite.resolutions) == 1
    assert len(suite.resolutions[0]) == 1
    resolution = suite.resolutions[0][0]
    assert isinstance(resolution, batched_fft.BatchedFFTSizeResolution)
    assert resolution.requested_memory_target_bytes == target_bytes
    panel_lines = resolution.panel_lines()
    assert panel_lines[0] == (
        f"requested_memory_target: {target_bytes:,} bytes"
    )
    assert len(panel_lines) == len(batched_fft._CASES) + 1
    assert len(suite.calls) == len(batched_fft._CASES)
    assert suite.info_names == [case.name for case in batched_fft._CASES]
    for (call_name, call_args), case in zip(
        suite.calls, batched_fft._CASES, strict=True
    ):
        expected_batch = max(
            1, target_bytes // (2 * case.transform_volume * case.itemsize)
        )
        assert call_name == "batched_fft"
        assert call_args[1] == case.dims
        assert call_args[2] == case.dtype_name
        assert call_args[3] == expected_batch
        assert call_args[4] == case.extent
        assert (
            batched_fft._estimate_case_working_set_bytes(case, expected_batch)
            <= target_bytes
        )
        assert any(
            line.startswith(f"{case.name} work_scale=1: ")
            and f"effective_target={target_bytes:,} bytes" in line
            for line in panel_lines[1:]
        )


def test_batched_fft_memory_target_work_rescale_uses_effective_target() -> (
    None
):
    batched_fft = _module("batched_fft_bench")
    suite = _make_recording_suite()
    target_bytes = 64 << 10
    request = _sizing().SizeRequest(
        memory_target_bytes=[target_bytes], rescale_by_work=[1024.0, 262144.0]
    )

    batched_fft.run_benchmarks(suite, request)

    assert len(suite.resolutions) == 1
    assert len(suite.resolutions[0]) == 1
    panel_lines = suite.resolutions[0][0].panel_lines()
    assert len(panel_lines) == 2 * len(batched_fft._CASES) + 1

    expected_batches = {
        "1d": [16, 4096],
        "2d": [16, 4096],
        "3d": [16, 4096],
        "2d_double": [8, 2048],
    }
    for case_index, case in enumerate(batched_fft._CASES):
        first_args = suite.calls[2 * case_index][1]
        second_args = suite.calls[2 * case_index + 1][1]
        assert [first_args[3], second_args[3]] == expected_batches[case.name]
        assert any(
            line.startswith(f"{case.name} work_scale=262144: ")
            and "effective_target=17,179,869,184 bytes" in line
            for line in panel_lines[1:]
        )


def test_batched_fft_effective_target_equality_does_not_warn() -> None:
    batched_fft = _module("batched_fft_bench")
    suite = _make_recording_suite()
    target_bytes = 64 << 10
    request = _sizing().SizeRequest(
        memory_target_bytes=[target_bytes], rescale_by_work=[128.0]
    )

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        batched_fft.run_benchmarks(suite, request)

    assert not recorded
    for case_index, case in enumerate(batched_fft._CASES):
        args = suite.calls[case_index][1]
        expected_batch = 2 if case.dtype_name == "complex64" else 1
        assert args[3] == expected_batch


def test_batched_fft_effective_target_equality_uses_strict_warning() -> None:
    batched_fft = _module("batched_fft_bench")
    suite = _make_recording_suite()
    target_bytes = 64 << 10
    request = _sizing().SizeRequest(
        memory_target_bytes=[target_bytes], rescale_by_work=[64.0]
    )

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        batched_fft.run_benchmarks(suite, request)

    assert len(recorded) == 1
    message = str(recorded[0].message)
    assert "2d_double work_scale=64=" in message
    assert "1d work_scale=64=" not in message
    assert "2d work_scale=64=" not in message
    assert "3d work_scale=64=" not in message
    for case_index in range(len(batched_fft._CASES)):
        args = suite.calls[case_index][1]
        assert args[3] == 1


def test_batched_fft_exact_size_scales_only_the_batch_dimension() -> None:
    batched_fft = _module("batched_fft_bench")
    suite = _make_recording_suite(forbid_resolution=True)
    request = _sizing().SizeRequest(
        exact_size=[3 * batched_fft._SHARED_TRANSFORM_VOLUME]
    )

    batched_fft.run_benchmarks(suite, request)

    assert len(suite.calls) == len(batched_fft._CASES)
    assert suite.info_names == [case.name for case in batched_fft._CASES]
    for (call_name, call_args), case in zip(
        suite.calls, batched_fft._CASES, strict=True
    ):
        assert call_name == "batched_fft"
        assert call_args[1] == case.dims
        assert call_args[2] == case.dtype_name
        assert call_args[3] == 3
        assert call_args[4] == case.extent


def test_solve_suite_plans_scalar_cases_with_work_rescale() -> None:
    solve_bench = _module("solve_bench")
    suite = _make_recording_suite(forbid_resolution=True)
    request = _sizing().SizeRequest(
        exact_size=[64], rescale_by_work=[1.0, 8.0]
    )

    solve_bench.run_benchmarks(
        suite, request, variant="solve-1-rhs", precision="32"
    )

    assert suite.info_names == ["solve"]
    assert len(suite.calls) == 2
    sizes = [call_args[2] for _, call_args in suite.calls]
    # size=64 gives n=8 and work=576; 8x work lands at size=256.
    assert sizes == [64, 256]
    assert [call.case_id for call in suite.call_records] == [
        "solve.solve-1-rhs.float32",
        "solve.solve-1-rhs.float32",
    ]


def test_solve_suite_exact_size_uses_literal_base_per_case() -> None:
    solve_bench = _module("solve_bench")
    suite = _make_recording_suite(forbid_resolution=True)
    sizing = _sizing()
    request = sizing.SizeRequest(exact_size=[64], rescale_by_work=[1.0, 8.0])

    solve_bench.run_benchmarks(suite, request, variant="all", precision="32")

    variants = solve_bench._get_variants("all")
    assert suite.info_names == ["solve"]
    assert len(suite.calls) == 2 * len(variants)
    for case_index, case_name in enumerate(variants):
        first_record = suite.call_records[2 * case_index]
        second_record = suite.call_records[2 * case_index + 1]
        expected_sizes = sizing.rescale_sizes_by_work(
            request,
            [64],
            estimate_work=lambda size, case_name=case_name: (
                solve_bench._estimate_case_work(case_name, size)
            ),
        )
        assert [first_record.args[2], second_record.args[2]] == expected_sizes
        assert first_record.args[1] == case_name
        assert first_record.case_id == f"solve.{case_name}.float32"


def test_solve_suite_resolves_memory_target_per_case() -> None:
    solve_bench = _module("solve_bench")
    suite = _make_recording_suite()
    sizing = _sizing()
    target_bytes = 1 << 20
    request = sizing.SizeRequest(
        memory_target_bytes=[target_bytes], rescale_by_work=[1.0, 4.0]
    )

    solve_bench.run_benchmarks(suite, request, variant="all", precision="32")

    variants = solve_bench._get_variants("all")
    assert suite.info_names == ["solve"]
    assert len(suite.calls) == 2 * len(variants)
    assert len(suite.resolutions) == 1
    assert len(suite.resolutions[0]) == len(variants)

    base_sizes = []
    for case_index, case_name in enumerate(variants):
        expected_base = solve_bench._resolve_case_size_from_memory_target(
            case_name, "float32", target_bytes
        )
        expected_sizes = sizing.rescale_sizes_by_work(
            request,
            [expected_base],
            estimate_work=lambda size, case_name=case_name: (
                solve_bench._estimate_case_work(case_name, size)
            ),
        )
        first_record = suite.call_records[2 * case_index]
        second_record = suite.call_records[2 * case_index + 1]
        assert [first_record.args[2], second_record.args[2]] == expected_sizes
        assert first_record.args[1] == case_name
        assert (
            solve_bench._estimate_case_working_set_bytes(
                case_name, "float32", expected_base
            )
            <= target_bytes
        )
        assert first_record.case_id == f"solve.{case_name}.float32"
        panel_lines = suite.resolutions[0][case_index].panel_lines()
        assert f"case: solve.{case_name}.float32" in panel_lines
        base_sizes.append(expected_base)

    assert len(set(base_sizes)) > 1


def test_gemm_gemv_suite_plans_named_cases_with_work_rescale() -> None:
    gemm_gemv = _module("gemm_gemv_bench")
    suite = _make_recording_suite(forbid_resolution=True)
    sizing = _sizing()
    request = sizing.SizeRequest(exact_size=[64], rescale_by_work=[1.0, 4.0])

    gemm_gemv.run_benchmarks(
        suite, request, variant="all", precision="32", perform_check=False
    )

    assert suite.info_names == ["skinny_gemm", "square_gemm", "gemv"]
    assert len(suite.calls) == 6
    for case_index, case_name in enumerate(suite.info_names):
        first_args = suite.calls[2 * case_index][1]
        second_args = suite.calls[2 * case_index + 1][1]
        expected_sizes = sizing.rescale_sizes_by_work(
            request,
            [64],
            estimate_work=lambda size, case_name=case_name: (
                gemm_gemv._estimate_case_work(case_name, size)
            ),
        )
        assert [first_args[1], second_args[1]] == expected_sizes
        assert suite.call_records[2 * case_index].case_id == (
            f"gemm_gemv.{case_name}.float32"
        )


def test_gemm_gemv_suite_resolves_memory_target_per_case() -> None:
    gemm_gemv = _module("gemm_gemv_bench")
    suite = _make_recording_suite()
    sizing = _sizing()
    target_bytes = 1 << 20
    request = sizing.SizeRequest(
        memory_target_bytes=[target_bytes], rescale_by_work=[1.0, 4.0]
    )

    gemm_gemv.run_benchmarks(
        suite, request, variant="all", precision="32", perform_check=False
    )

    assert suite.info_names == ["skinny_gemm", "square_gemm", "gemv"]
    assert len(suite.calls) == 6
    assert len(suite.resolutions) == 1
    assert len(suite.resolutions[0]) == 3

    base_sizes = []
    for case_index, case_name in enumerate(suite.info_names):
        expected_base = gemm_gemv._resolve_case_size_from_memory_target(
            case_name, "float32", target_bytes
        )
        expected_sizes = sizing.rescale_sizes_by_work(
            request,
            [expected_base],
            estimate_work=lambda size, case_name=case_name: (
                gemm_gemv._estimate_case_work(case_name, size)
            ),
        )
        first_args = suite.calls[2 * case_index][1]
        second_args = suite.calls[2 * case_index + 1][1]
        assert [first_args[1], second_args[1]] == expected_sizes
        assert (
            gemm_gemv._estimate_case_working_set_bytes(
                case_name, "float32", expected_base
            )
            <= target_bytes
        )
        assert suite.call_records[2 * case_index].case_id == (
            f"gemm_gemv.{case_name}.float32"
        )
        panel_lines = suite.resolutions[0][case_index].panel_lines()
        assert f"case: gemm_gemv.{case_name}.float32" in panel_lines
        base_sizes.append(expected_base)

    assert len(set(base_sizes)) > 1


def test_axis_sum_suite_plans_generated_cases_with_work_rescale() -> None:
    axis_sum = _module("axis_sum_bench")
    suite = _make_recording_suite(forbid_resolution=True)
    request = _sizing().SizeRequest(
        exact_size=[100], rescale_by_work=[1.0, 2.0]
    )

    axis_sum.run_benchmarks(suite, request, case="all", perform_check=False)

    assert suite.info_names == [*axis_sum._CASES]
    assert len(suite.calls) == 2 * len(axis_sum._CASES)
    for case_index, case_name in enumerate(axis_sum._CASES):
        first_args = suite.calls[2 * case_index][1]
        second_args = suite.calls[2 * case_index + 1][1]
        assert first_args[1] == case_name
        assert first_args[6] == 100
        assert second_args[6] > first_args[6]
        assert suite.call_records[2 * case_index].case_id == (
            f"axis_sum.{case_name}"
        )


def test_batched_fft_suite_plans_per_case_batches_with_work_rescale() -> None:
    batched_fft = _module("batched_fft_bench")
    suite = _make_recording_suite(forbid_resolution=True)
    request = _sizing().SizeRequest(
        exact_size=[2 * batched_fft._SHARED_TRANSFORM_VOLUME],
        rescale_by_work=[1.0, 2.0],
    )

    batched_fft.run_benchmarks(suite, request)

    assert suite.info_names == [case.name for case in batched_fft._CASES]
    assert len(suite.calls) == 2 * len(batched_fft._CASES)
    for case_index, case in enumerate(batched_fft._CASES):
        first_args = suite.calls[2 * case_index][1]
        second_args = suite.calls[2 * case_index + 1][1]
        assert first_args[1] == case.dims
        assert first_args[2] == case.dtype_name
        assert [first_args[3], second_args[3]] == [2, 4]
        assert suite.call_records[2 * case_index].case_id == (
            f"batched_fft.{case.name}"
        )


def test_stream_suite_supports_work_rescale() -> None:
    stream_bench = _module("stream_bench")
    suite = _make_recording_suite(forbid_resolution=True)
    request = _sizing().SizeRequest(
        exact_size=[100], rescale_by_work=[1.0, 2.0]
    )

    stream_bench.run_benchmarks(
        suite,
        request,
        operation="copy",
        precision="32",
        contiguous="true",
        perform_check=False,
    )

    assert suite.calls
    assert suite.calls[0][1][4] == [100, 200]


def test_sort_suite_supports_work_rescale() -> None:
    sort_bench = _module("sort_bench")
    suite = _make_recording_suite(forbid_resolution=True)
    request = _sizing().SizeRequest(
        exact_size=[100], rescale_by_work=[1.0, 2.0]
    )

    sort_bench.run_benchmarks(
        suite, request, variant="sort-1D", precision="32"
    )

    assert suite.calls
    sizes = suite.calls[0][1][2]
    assert sizes[0] == 100
    assert sizes[1] > sizes[0]


def test_resolve_linear_suite_size_reports_resolution_details() -> None:
    sizing = _sizing()
    sizes, resolutions = sizing.resolve_linear_suite_size(
        sizing.SizeRequest(memory_target_bytes=[100]),
        bytes_per_element=9,
        describe_size=lambda resolved_size: [f"shape: {resolved_size}"],
    )

    assert sizes == [11]
    assert resolutions is not None
    assert len(resolutions) == 1
    assert resolutions[0].requested_memory_target_bytes == 100
    assert resolutions[0].estimated_working_set_bytes == 99
    assert resolutions[0].detail_lines == ("shape: 11",)


def test_resolve_size_by_binary_search_finds_largest_fitting_size() -> None:
    sizing = _sizing()

    def estimate(size: int) -> int:
        return size * size + size

    _assert_target_resolution(
        lambda target_bytes: sizing.resolve_size_by_binary_search(
            target_bytes,
            estimate_working_set_bytes=estimate,
            initial_guess=target_bytes // 4,
        ),
        estimate,
        target_bytes=1_000,
    )


def test_resolve_suite_size_rescales_exact_size_by_work() -> None:
    sizing = _sizing()
    request = sizing.SizeRequest(
        exact_size=[100], rescale_by_work=[1.0, 4.0, 0.25]
    )

    sizes, resolutions = sizing.resolve_suite_size(
        request,
        resolve_from_target=lambda target_bytes: target_bytes,
        estimate_working_set_bytes=lambda size: size,
        estimate_work=lambda size: size * size,
    )

    assert sizes == [100, 200, 50]
    assert resolutions is None


def test_resolve_suite_size_rescales_each_memory_target_by_work() -> None:
    sizing = _sizing()
    request = sizing.SizeRequest(
        memory_target_bytes=[10, 20], rescale_by_work=[1.0, 2.0]
    )

    sizes, resolutions = sizing.resolve_suite_size(
        request,
        resolve_from_target=lambda target_bytes: target_bytes,
        estimate_working_set_bytes=lambda size: size,
        estimate_work=lambda size: size,
    )

    assert sizes == [10, 20, 20, 40]
    assert resolutions is not None
    assert [r.resolved_size for r in resolutions] == [10, 20]
    assert [r.requested_memory_target_bytes for r in resolutions] == [10, 20]
    assert resolutions[0].panel_lines()[-3:] == [
        "work-rescaled sizes:",
        "work_scale=1: resolved_size=10",
        "work_scale=2: resolved_size=20",
    ]


def test_resolve_suite_size_rejects_rescale_without_work_estimate() -> None:
    sizing = _sizing()
    request = sizing.SizeRequest(exact_size=[100], rescale_by_work=[1.0, 2.0])

    with pytest.raises(RuntimeError, match="work rescaling"):
        sizing.resolve_suite_size(
            request,
            resolve_from_target=lambda target_bytes: target_bytes,
            estimate_working_set_bytes=lambda size: size,
        )


def test_resolve_suite_size_warns_when_work_rescale_clamps_to_minimum() -> (
    None
):
    sizing = _sizing()
    request = sizing.SizeRequest(exact_size=[1], rescale_by_work=[0.25])

    with pytest.warns(
        RuntimeWarning,
        match="work target is smaller than minimum estimated work",
    ):
        sizes, resolutions = sizing.resolve_suite_size(
            request,
            resolve_from_target=lambda target_bytes: target_bytes,
            estimate_working_set_bytes=lambda size: size,
            estimate_work=lambda size: size + 10,
        )

    assert sizes == [1]
    assert resolutions is None


def test_resolve_suite_size_warns_before_rescaling_undersized_target() -> None:
    sizing = _sizing()
    request = sizing.SizeRequest(
        memory_target_bytes=[1], rescale_by_work=[1.0, 2.0]
    )

    with pytest.warns(
        RuntimeWarning,
        match="memory target is smaller than estimated working set",
    ):
        sizes, resolutions = sizing.resolve_suite_size(
            request,
            resolve_from_target=lambda target_bytes: target_bytes,
            estimate_working_set_bytes=lambda size: 10 * size,
            estimate_work=lambda size: size,
        )

    assert sizes == [1, 2]
    assert resolutions is not None
    assert resolutions[0].estimated_working_set_bytes == 10


def test_stream_target_resolution_for_noncontiguous_layout() -> None:
    stream_bench = _module("stream_bench")
    target_bytes = 1 << 20
    size = stream_bench._resolve_size_from_memory_target(
        "all", "all", target_bytes
    )

    side = math.isqrt(size)
    assert side * side == size
    assert stream_bench.get_noncontiguous_shape(size) == (side, side)
    assert (
        stream_bench._estimate_working_set_bytes("all", size) <= target_bytes
    )

    next_size = (side + 1) * (side + 1)
    assert (
        stream_bench._estimate_working_set_bytes("all", next_size)
        > target_bytes
    )


def test_stream_noncontiguous_initializer_uses_random_final_shape() -> None:
    stream_bench = _module("stream_bench")
    random_array = _TransposeStub("a")
    full_b = _TransposeStub("b")
    full_c = _TransposeStub("c")
    generator = SimpleNamespace(random=Mock(return_value=random_array))
    tracker_float32 = object()
    tracker = SimpleNamespace(
        float32=tracker_float32,
        arange=Mock(
            side_effect=AssertionError("non-contiguous init should not arange")
        ),
        full=Mock(side_effect=[full_b, full_c]),
        random=SimpleNamespace(default_rng=Mock(return_value=generator)),
    )

    result = stream_bench.initialize(tracker, 36, "float32", False)
    shape = stream_bench.get_noncontiguous_shape(36)

    tracker.random.default_rng.assert_called_once_with()
    generator.random.assert_called_once_with(shape, dtype=tracker_float32)
    assert tracker.full.call_args_list == [
        ((shape, 2), {"dtype": "float32"}),
        ((shape, 1), {"dtype": "float32"}),
    ]
    assert result == (random_array.T, full_b.T, full_c.T)


@pytest.mark.parametrize("operation", ["copy", "mul", "add"])
def test_stream_check_uses_actual_initialized_arrays(operation: str) -> None:
    stream_bench = _module("stream_bench")
    a = np.array([[0.125, 0.25], [0.5, 0.75]], dtype=np.float32).T
    b = np.full_like(a, 2)
    c = np.full_like(a, 1)

    if operation == "copy":
        result = a.copy()
    elif operation == "mul":
        result = c * stream_bench.SCALAR
    else:
        result = a + b

    stream_bench.check_stream(operation, a, b, c, result)


def test_stream_check_uses_pre_op_snapshots(monkeypatch) -> None:
    stream_bench = _module("stream_bench")
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.full_like(a, 2.0)
    c = np.full_like(a, 1.0)

    def fake_initialize(np_module, size, dtype, contiguous):
        del np_module, size, dtype, contiguous
        return a, b, c

    def fake_timed_loop(op, timer, runs, warmup):
        del op, timer, runs, warmup
        a[...] = a + 1.0
        c[...] = a
        return 1.0

    monkeypatch.setattr(stream_bench, "initialize", fake_initialize)
    monkeypatch.setattr(stream_bench, "timed_loop", fake_timed_loop)

    with pytest.raises(AssertionError, match="stream result mismatch"):
        stream_bench.stream(
            np,
            "copy",
            True,
            "float32",
            3,
            1,
            0,
            timer=object(),
            perform_check=True,
        )


@pytest.mark.parametrize(
    ("variant", "dtype"),
    [
        ("skinny_gemm", "float32"),
        ("skinny_gemm", "float64"),
        ("square_gemm", "float32"),
        ("square_gemm", "float64"),
        ("gemv", "float32"),
        ("gemv", "float64"),
    ],
)
def test_gemm_target_resolution_for_each_case(
    variant: str, dtype: str
) -> None:
    gemm_gemv = _module("gemm_gemv_bench")

    def estimate(size: int) -> int:
        return gemm_gemv._estimate_case_working_set_bytes(variant, dtype, size)

    _assert_target_resolution(
        lambda target_bytes: (
            gemm_gemv._resolve_case_size_from_memory_target(
                variant, dtype, target_bytes
            )
        ),
        estimate,
        target_bytes=8 << 20,
    )


@pytest.mark.parametrize(
    "variant",
    [
        "solve-1-rhs",
        "solve-n-rhs",
        "batched-solve-1-rhs",
        "batched-solve-n-rhs",
    ],
)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_solve_target_resolution_for_each_case(
    variant: str, dtype: str
) -> None:
    solve_bench = _module("solve_bench")

    def estimate(size: int) -> int:
        return solve_bench._estimate_case_working_set_bytes(
            variant, dtype, size
        )

    _assert_target_resolution(
        lambda target_bytes: (
            solve_bench._resolve_case_size_from_memory_target(
                variant, dtype, target_bytes
            )
        ),
        estimate,
        target_bytes=8 << 20,
    )


def test_undersized_stream_target_raises() -> None:
    sizing = _sizing()
    stream_bench = _module("stream_bench")
    with pytest.warns(
        RuntimeWarning,
        match="memory target is smaller than estimated working set",
    ):
        _, resolutions = sizing.resolve_suite_size(
            sizing.SizeRequest(memory_target_bytes=[1]),
            resolve_from_target=lambda target_bytes: (
                stream_bench._resolve_size_from_memory_target(
                    "all", "false", target_bytes
                )
            ),
            estimate_working_set_bytes=lambda size: (
                stream_bench._estimate_working_set_bytes("all", size)
            ),
        )
    assert resolutions is not None
    assert len(resolutions) == 1
    assert resolutions[0].estimated_working_set_bytes > 1


def test_batched_fft_undersized_target_warns() -> None:
    batched_fft = _module("batched_fft_bench")
    suite = _make_recording_suite()
    request = _sizing().SizeRequest(
        memory_target_bytes=[100], rescale_by_work=[1.0, 2.0]
    )

    with pytest.warns(
        RuntimeWarning,
        match="memory target is smaller than estimated working set for "
        "batched FFT case\\(s\\)",
    ):
        batched_fft.run_benchmarks(suite, request)

    assert len(suite.resolutions) == 1
    assert len(suite.resolutions[0]) == 1
    panel_lines = suite.resolutions[0][0].panel_lines()
    assert (
        "1d work_scale=1: effective_target=100 bytes, batch=1, "
        in (panel_lines[1])
    )
    assert (
        "1d work_scale=2: effective_target=200 bytes, batch=1, "
        in (panel_lines[2])
    )
    assert len(suite.calls) == 2 * len(batched_fft._CASES)
    for case_index, case in enumerate(batched_fft._CASES):
        first_args = suite.calls[2 * case_index][1]
        second_args = suite.calls[2 * case_index + 1][1]
        assert first_args[1] == case.dims
        assert [first_args[3], second_args[3]] == [1, 1]


def test_fast_advanced_indexing_uses_square_size_for_2d_cases() -> None:
    fast_advanced_indexing = _module("fast_advanced_indexing_bench")
    suite = _make_recording_suite(forbid_resolution=True)

    fast_advanced_indexing.run_benchmarks(
        suite, _sizing().SizeRequest(exact_size=[10_000])
    )

    call_map = {name: args for name, args in suite.calls}
    assert call_map["putmask_scalar"][1] == [10_000]
    for name in ("take_2d",):
        assert call_map[name][1] == 100


def test_fast_advanced_indexing_clamps_small_targets_to_nonzero_indices() -> (
    None
):
    fast_advanced_indexing = _module("fast_advanced_indexing_bench")
    suite = _make_recording_suite()

    fast_advanced_indexing.run_benchmarks(
        suite, _sizing().SizeRequest(memory_target_bytes=[100])
    )

    call_map = {name: args for name, args in suite.calls}
    for name in ("take_2d",):
        assert call_map[name][2] > 0


def test_general_indexing_clamps_small_targets_to_nonzero_indices() -> None:
    general_indexing = _module("general_indexing_bench")
    suite = _make_recording_suite()

    general_indexing.run_benchmarks(
        suite, _sizing().SizeRequest(memory_target_bytes=[100])
    )

    call_map = {name: args for name, args in suite.calls}
    for name in (
        "mixed_indexing",
        "non_contiguous_indexing",
        "array_get_1d",
        "array_set_1d",
        "scalar_list_set_2d",
    ):
        assert call_map[name][2] > 0


def test_axis_sum_normalizes_negative_axes_for_output_shape() -> None:
    axis_sum = _module("axis_sum_bench")

    assert axis_sum._normalized_axes(-1, 3) == (2,)
    assert axis_sum._normalized_axes((0, -1), 3) == (0, 2)


def test_main_dispatches_memory_target_request(monkeypatch) -> None:
    main = _module("main")
    requests = []

    class FakeSuite:
        name = "fake"

        @staticmethod
        def add_suite_parser_group(parser) -> None:
            del parser

        def __init__(self, config, args) -> None:
            del config, args
            self.benchmark_count = 1

        def __enter__(self):
            return self

        def __exit__(self, *exc_info) -> None:
            del exc_info

        def run_suite(self, size_request) -> None:
            requests.append(size_request)

    config = SimpleNamespace(
        summarize=None,
        summarize_flush=main.SummarizeFlush.NEVER,
        repeat=0,
        package="numpy",
        runs=1,
        warmup=0,
        print_panel=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        main.MicrobenchmarkConfig,
        "add_parser_group",
        lambda parser, name: None,
    )
    monkeypatch.setattr(
        main.MicrobenchmarkConfig, "from_args", lambda args: config
    )
    monkeypatch.setattr(main, "SUITE_CLASSES", [FakeSuite])

    assert main.main(["--suite", "fake", "--memory-size", "64MiB"]) == 0
    assert len(requests) == 1
    assert requests[0].memory_target_bytes == [64 << 20]


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))

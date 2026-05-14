from __future__ import annotations

import argparse
import math
import re
import warnings

from argparse import ArgumentParser, Namespace
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace

DEFAULT_PROBLEM_SIZE = 10_000_000

_MEMORY_SIZE_PATTERN = re.compile(
    r"^(?P<value>[0-9]+)(?P<unit>B|KiB|MiB|GiB|TiB)$"
)
_MEMORY_SIZE_UNITS = {
    "B": 1,
    "KiB": 1 << 10,
    "MiB": 1 << 20,
    "GiB": 1 << 30,
    "TiB": 1 << 40,
}


def parse_memory_size(value: str) -> int:
    match = _MEMORY_SIZE_PATTERN.fullmatch(value.strip())
    if match is None:
        raise argparse.ArgumentTypeError(
            "memory size must use <integer><unit> with "
            "B, KiB, MiB, GiB, or TiB"
        )

    amount = int(match.group("value"))
    if amount <= 0:
        raise argparse.ArgumentTypeError("memory size must be positive")

    unit = match.group("unit")
    return amount * _MEMORY_SIZE_UNITS[unit]


def parse_work_scale(value: str) -> float:
    try:
        amount = float(value)
    except ValueError as ex:
        raise argparse.ArgumentTypeError(
            "work scale must be a positive finite number"
        ) from ex

    if not math.isfinite(amount) or amount <= 0.0:
        raise argparse.ArgumentTypeError(
            "work scale must be a positive finite number"
        )
    return amount


def clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def add_size_request_parser_group(parser: ArgumentParser) -> None:
    group = parser.add_argument_group()
    group.add_argument(
        "--rescale-by-work",
        dest="rescale_by_work",
        metavar="WORK",
        type=parse_work_scale,
        nargs="+",
        default=argparse.SUPPRESS,
        help="Relative work factors to apply to each base size",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--size",
        type=int,
        default=argparse.SUPPRESS,
        nargs="+",
        help=(
            "Exact benchmark size with suite-specific semantics "
            f"(default: {DEFAULT_PROBLEM_SIZE:,} when neither sizing flag "
            "is provided)"
        ),
    )
    group.add_argument(
        "--memory-size",
        dest="memory_size",
        metavar="SIZE",
        type=parse_memory_size,
        default=argparse.SUPPRESS,
        nargs="+",
        help=(
            "Approximate benchmark working-set target using binary units "
            "(B, KiB, MiB, GiB, TiB)"
        ),
    )


@dataclass(frozen=True)
class SizeResolution:
    resolved_size: int
    requested_memory_target_bytes: int
    estimated_working_set_bytes: int
    detail_lines: tuple[str, ...] = ()

    def panel_lines(self) -> list[str]:
        lines = [
            f"resolved_size: {self.resolved_size:,}",
            (
                "requested_memory_target: "
                f"{self.requested_memory_target_bytes:,} bytes"
            ),
            (
                "estimated_working_set: "
                f"{self.estimated_working_set_bytes:,}"
                " bytes"
            ),
        ]
        lines.extend(self.detail_lines)
        return lines


@dataclass(frozen=True)
class SizeRequest:
    exact_size: list[int] | None = None
    memory_target_bytes: list[int] | None = None
    rescale_by_work: list[float] = field(default_factory=lambda: [1.0])

    def __post_init__(self) -> None:
        if not self.rescale_by_work:
            raise RuntimeError(
                "--rescale-by-work must specify at least one value"
            )
        if any(
            (not math.isfinite(work_scale)) or work_scale <= 0.0
            for work_scale in self.rescale_by_work
        ):
            raise RuntimeError("--rescale-by-work values must be positive")

    @classmethod
    def from_namespace(
        cls,
        args: Namespace,
        *,
        default_exact_size: int | None = DEFAULT_PROBLEM_SIZE,
    ) -> SizeRequest:
        exact_size = getattr(args, "size", None)
        memory_target_bytes = getattr(args, "memory_size", None)
        rescale_by_work = getattr(args, "rescale_by_work", [1.0])

        if exact_size is None and memory_target_bytes is None:
            if default_exact_size is None:
                raise RuntimeError("size request must specify a sizing mode")
            exact_size = [default_exact_size]

        if exact_size is not None and memory_target_bytes is not None:
            raise RuntimeError(
                "--size and --memory-size must be mutually exclusive"
            )
        return cls(
            exact_size=exact_size,
            memory_target_bytes=memory_target_bytes,
            rescale_by_work=rescale_by_work,
        )

    @property
    def uses_work_rescale(self) -> bool:
        return len(self.rescale_by_work) != 1 or self.rescale_by_work[0] != 1.0

    def config_lines(self) -> list[str]:
        if self.exact_size is not None:
            sizes = ", ".join([f"{e:,}" for e in self.exact_size])
            lines = ["Sizing: exact (--size)", f"Suite-defined size: {sizes}"]
            if self.uses_work_rescale:
                scales = ", ".join(str(e) for e in self.rescale_by_work)
                lines.append(f"Work rescale factors: {scales}")
            return lines

        assert self.memory_target_bytes is not None
        sizes = ", ".join([f"{e:,}" for e in self.memory_target_bytes])
        lines = [
            "Sizing: working-set target (--memory-size)",
            (f"Approximate working-set target (bytes): {sizes}"),
        ]
        if self.uses_work_rescale:
            scales = ", ".join(str(e) for e in self.rescale_by_work)
            lines.append(f"Work rescale factors: {scales}")
        return lines


def resolve_size_by_monotonic_search(
    target: int | float,
    *,
    estimate_value: Callable[[int], int | float],
    initial_guess: int,
) -> int:
    """
    Find the largest size whose monotonic estimate is less than or equal to
    a target. If even the minimum size exceeds the target, return the minimum
    size and let the caller decide whether to warn or fail.
    """
    if target < 0:
        raise ValueError("target must be non-negative")

    low = 1
    low_value = estimate_value(low)
    if low_value > target:
        return low

    high = max(2, initial_guess)
    high_value = estimate_value(high)
    if high_value < low_value:
        raise RuntimeError("Binary search function violates monotonicity")

    for _ in range(64):
        if high_value > target:
            break
        low = high
        low_value = high_value
        high *= 2
        new_high_value = estimate_value(high)
        if new_high_value < high_value:
            raise RuntimeError("Binary search function violates monotonicity")
        high_value = new_high_value
    else:
        raise RuntimeError("Unable to find upper bound for binary search")

    while low + 1 < high:
        mid = low + (high - low) // 2
        mid_value = estimate_value(mid)
        if mid_value < low_value or mid_value > high_value:
            raise RuntimeError("Binary search function violates monotonicity")
        if mid_value <= target:
            low = mid
            low_value = mid_value
        else:
            high = mid
            high_value = mid_value
    return low


def rescale_sizes_by_work(
    size_request: SizeRequest,
    base_sizes: Iterable[int],
    *,
    estimate_work: Callable[[int], int | float] | None,
) -> list[int]:
    if not size_request.uses_work_rescale:
        return [*base_sizes]
    if estimate_work is None:
        raise RuntimeError("work rescaling is not supported for this suite")

    sizes: list[int] = []
    for base_size in base_sizes:
        base_work = estimate_work(base_size)
        if base_work <= 0:
            raise RuntimeError("work estimate must be positive")
        for work_scale in size_request.rescale_by_work:
            if work_scale == 1.0:
                sizes.append(base_size)
                continue
            target_work = base_work * work_scale
            resolved_size = resolve_size_by_monotonic_search(
                target_work,
                estimate_value=estimate_work,
                initial_guess=max(1, int(base_size * work_scale)),
            )
            resolved_work = estimate_work(resolved_size)
            if resolved_work > target_work:
                warnings.warn(
                    "work target is smaller than minimum estimated work: "
                    f"estimated={resolved_work}, target={target_work}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            sizes.append(resolved_size)
    return sizes


def _work_rescale_detail_lines(
    size_request: SizeRequest, sizes: Iterable[int]
) -> tuple[str, ...]:
    if not size_request.uses_work_rescale:
        return ()

    return (
        "work-rescaled sizes:",
        *(
            f"work_scale={work_scale:g}: resolved_size={size:,}"
            for work_scale, size in zip(
                size_request.rescale_by_work, sizes, strict=True
            )
        ),
    )


def resolve_suite_size(
    size_request: SizeRequest,
    *,
    resolve_from_target: Callable[[int], int],
    estimate_working_set_bytes: Callable[[int], int],
    estimate_work: Callable[[int], int | float] | None = None,
    describe_size: Callable[[int], Iterable[str]] | None = None,
) -> tuple[list[int], list[SizeResolution] | None]:
    """
    Resolve an exact suite size from an explicit size or memory target.

    Parameters
    ----------
    size_request : SizeRequest
        User-provided sizing mode, target values, and work rescale factors.
    resolve_from_target : Callable[[int], int]
        Maps a target working-set size in bytes to a suite-specific size.
    estimate_working_set_bytes : Callable[[int], int]
        Estimates the suite working set for a resolved size.
    estimate_work : Callable[[int], int | float] | None, optional
        Estimates suite work for a resolved size. Required when
        ``size_request`` asks for work rescaling.
    describe_size : Callable[[int], Iterable[str]] | None, optional
        Produces additional human-readable sizing details.
    """
    if size_request.exact_size is not None:
        return (
            rescale_sizes_by_work(
                size_request,
                size_request.exact_size,
                estimate_work=estimate_work,
            ),
            None,
        )

    resolved_sizes = []
    resolutions = []
    assert size_request.memory_target_bytes is not None
    for target_bytes in size_request.memory_target_bytes:
        resolved_size = resolve_from_target(target_bytes)
        detail_lines: Iterable[str] = ()
        if describe_size is not None:
            detail_lines = describe_size(resolved_size)
        estimated_working_set_bytes = estimate_working_set_bytes(resolved_size)
        resolution = SizeResolution(
            resolved_size=resolved_size,
            requested_memory_target_bytes=target_bytes,
            estimated_working_set_bytes=estimated_working_set_bytes,
            detail_lines=tuple(detail_lines),
        )
        resolved_sizes.append(resolved_size)
        resolutions.append(resolution)
        if estimated_working_set_bytes > target_bytes:
            warnings.warn(
                "memory target is smaller than estimated working set: "
                f"estimated={estimated_working_set_bytes:,} bytes, "
                f"target={target_bytes:,} bytes",
                RuntimeWarning,
                stacklevel=2,
            )
    rescaled_sizes = rescale_sizes_by_work(
        size_request, resolved_sizes, estimate_work=estimate_work
    )
    if size_request.uses_work_rescale:
        scale_count = len(size_request.rescale_by_work)
        resolutions = [
            replace(
                resolution,
                detail_lines=(
                    *resolution.detail_lines,
                    *_work_rescale_detail_lines(
                        size_request,
                        rescaled_sizes[
                            index * scale_count : (index + 1) * scale_count
                        ],
                    ),
                ),
            )
            for index, resolution in enumerate(resolutions)
        ]
    return (rescaled_sizes, resolutions)


def resolve_size_by_binary_search(
    target_bytes: int,
    *,
    estimate_working_set_bytes: Callable[[int], int],
    initial_guess: int,
) -> int:
    return resolve_size_by_monotonic_search(
        target_bytes,
        estimate_value=estimate_working_set_bytes,
        initial_guess=initial_guess,
    )


def resolve_linear_suite_size(
    size_request: SizeRequest,
    *,
    bytes_per_element: int,
    describe_size: Callable[[int], Iterable[str]] | None = None,
) -> tuple[list[int], list[SizeResolution] | None]:
    if bytes_per_element <= 0:
        raise ValueError("bytes_per_element must be positive")

    return resolve_suite_size(
        size_request,
        resolve_from_target=lambda target_bytes: max(
            1, target_bytes // bytes_per_element
        ),
        estimate_working_set_bytes=lambda resolved_size: (
            bytes_per_element * resolved_size
        ),
        estimate_work=lambda resolved_size: resolved_size,
        describe_size=describe_size,
    )

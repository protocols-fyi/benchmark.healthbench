"""Shared validation helpers for benchmark models."""

from collections.abc import Sequence


def strip_and_require_non_empty(value: str, *, message: str) -> str:
    value = value.strip()
    assert value, message
    return value


def strip_optional_non_empty(value: str | None, *, message: str) -> str | None:
    if value is None:
        return None
    value = value.strip()
    assert value, message
    return value


def strip_string(value: str) -> str:
    return value.strip()


def normalize_string_tuple(value: object, *, message: str) -> tuple[str, ...]:
    if value is None:
        return ()
    assert isinstance(value, Sequence) and not isinstance(value, str), message
    return tuple(str(item) for item in value)


def require_non_zero_float(value: float, *, message: str) -> float:
    assert value != 0.0, message
    return value


def require_positive_int(value: int, *, message: str) -> int:
    assert value > 0, message
    return value


def require_non_negative_int(value: int, *, message: str) -> int:
    assert value >= 0, message
    return value


def require_int_choice(
    value: int,
    *,
    choices: frozenset[int],
    message: str,
) -> int:
    assert value in choices, message
    return value


def require_float_in_range(
    value: float,
    *,
    minimum: float,
    maximum: float,
    message: str,
) -> float:
    assert minimum <= value <= maximum, message
    return value

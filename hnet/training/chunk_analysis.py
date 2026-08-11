import math
from statistics import mean, median


def percentile(values: list[int], quantile: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_lengths(chunks: list[list[int]]) -> dict[str, float | int]:
    lengths = [len(chunk) for chunk in chunks]
    if not lengths:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "max": 0,
        }
    return {
        "count": len(lengths),
        "mean": mean(lengths),
        "median": median(lengths),
        "p50": percentile(lengths, 0.50),
        "p90": percentile(lengths, 0.90),
        "p95": percentile(lengths, 0.95),
        "max": max(lengths),
    }


def mid_utf8_boundary_rate(token_ids: list[int], positions: list[int]) -> float:
    continuation_positions = {
        index for index, token_id in enumerate(token_ids) if 0x80 <= token_id <= 0xBF
    }
    if not continuation_positions:
        return 0.0
    midpoints = continuation_positions.intersection(positions)
    return len(midpoints) / len(continuation_positions)


def target_boundary_offsets(
    prompt: str,
    target: str,
    boundary_positions: list[int],
    add_bos: bool = True,
) -> list[list[int]]:
    prompt_bytes = prompt.encode("utf-8")
    target_bytes = target.encode("utf-8")
    offsets: list[list[int]] = []
    cursor = 0
    bos_offset = 1 if add_bos else 0
    while True:
        start = prompt_bytes.find(target_bytes, cursor)
        if start < 0:
            break
        end = start + len(target_bytes)
        offsets.append(
            [
                position - bos_offset - start
                for position in boundary_positions
                if start + bos_offset <= position < end + bos_offset
            ]
        )
        cursor = start + 1
    return offsets

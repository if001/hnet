import json
import re
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = {
    "k1first_mix": REPOSITORY_ROOT / "configs/hnet_2stage_200m_k1first_mix.json",
    "k3first_mix": REPOSITORY_ROOT / "configs/hnet_2stage_200m_k3first_mix.json",
}
SCREENING_CONFIGS = {
    "k14g12_front": REPOSITORY_ROOT / "configs/hnet_2stage_200m_k14g12_front.json",
    "k14g12_middle": REPOSITORY_ROOT / "configs/hnet_2stage_200m_k14g12_middle.json",
    "k14g12_late": REPOSITORY_ROOT / "configs/hnet_2stage_200m_k14g12_late.json",
    "k15g11_split": REPOSITORY_ROOT / "configs/hnet_2stage_200m_k15g11_split.json",
    "k16g10_even": REPOSITORY_ROOT / "configs/hnet_2stage_200m_k16g10_even.json",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _main_layout(config: dict) -> str:
    return config["arch_layout"][1][1][0]


def _layer_counts(layout: str) -> dict[str, int]:
    parsed = re.findall(r"([KGT])(\d+)", layout)
    assert "".join(f"{kind}{count}" for kind, count in parsed) == layout
    return {
        kind: sum(int(count) for parsed_kind, count in parsed if parsed_kind == kind)
        for kind in "KGT"
    }


def test_mixed_configs_have_matched_layer_multisets() -> None:
    configs = {name: _load(path) for name, path in CONFIGS.items()}
    for config in configs.values():
        counts = _layer_counts(_main_layout(config))
        assert counts == {"K": 16, "G": 10, "T": 0}
        assert sum(counts.values()) == 26

    left = configs["k1first_mix"]
    right = configs["k3first_mix"]
    assert {**left, "arch_layout": None} == {**right, "arch_layout": None}


def test_mixed_configs_use_the_planned_order() -> None:
    layouts = {name: _main_layout(_load(path)) for name, path in CONFIGS.items()}
    assert layouts["k1first_mix"] == "K1G1" * 7 + "K3G1" * 3
    assert layouts["k3first_mix"] == "K3G1" * 3 + "K1G1" * 7


def test_factorized_screening_configs_use_planned_counts_and_order() -> None:
    layouts = {name: _main_layout(_load(path)) for name, path in SCREENING_CONFIGS.items()}
    assert layouts == {
        "k14g12_front": "K3G1" + "K1G1" * 11,
        "k14g12_middle": "K1G1" * 5 + "K3G1" + "K1G1" * 6,
        "k14g12_late": "K1G1" * 11 + "K3G1",
        "k15g11_split": "K3G1" + "K1G1" * 9 + "K3G1",
        "k16g10_even": (
            "K1G1" * 2
            + "K3G1"
            + "K1G1" * 2
            + "K3G1"
            + "K1G1" * 2
            + "K3G1"
            + "K1G1"
        ),
    }
    expected = {
        "k14g12_front": {"K": 14, "G": 12, "T": 0},
        "k14g12_middle": {"K": 14, "G": 12, "T": 0},
        "k14g12_late": {"K": 14, "G": 12, "T": 0},
        "k15g11_split": {"K": 15, "G": 11, "T": 0},
        "k16g10_even": {"K": 16, "G": 10, "T": 0},
    }
    for name, layout in layouts.items():
        counts = _layer_counts(layout)
        assert counts == expected[name]
        assert sum(counts.values()) == 26

    reference = _load(REPOSITORY_ROOT / "configs/hnet_2stage_200m_k1g1.json")
    for path in SCREENING_CONFIGS.values():
        candidate = _load(path)
        assert {**candidate, "arch_layout": None} == {
            **reference,
            "arch_layout": None,
        }

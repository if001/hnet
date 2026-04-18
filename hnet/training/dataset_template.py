from hnet.training import DatasetSource

## byte換算で0.5B token
SOURCES_JA8_EN1_CODE1: list[DatasetSource] = [
    DatasetSource(
        name="hotchpotch/fineweb-2-edu-japanese",
        config_name="small_tokens_cleaned",
        split="train",
        take_examples=68_000,
        skip_examples=10_000,  ## 先頭10000件スキップが推奨
    ),
    DatasetSource(
        name="wikimedia/wikipedia",
        config_name="20231101.ja",
        split="train",
        take_examples=12_000,
    ),
    DatasetSource(
        name="HuggingFaceFW/fineweb-edu",
        config_name="sample-10BT",
        split="train",
        take_examples=8_000,
    ),
    DatasetSource(
        name="wikimedia/wikipedia",
        config_name="20231101.en",
        split="train",
        take_examples=2_000,
    ),
    DatasetSource(
        name="codeparrot/codeparrot-clean",
        split="train",
        take_examples=10_000,
    ),
]


# 9:0:1
SOURCES_JA9_EN0_CODE1: list[DatasetSource] = [
    DatasetSource(
        name="hotchpotch/fineweb-2-edu-japanese",
        config_name="small_tokens_cleaned",
        split="train",
        take_examples=76_500,
        skip_examples=10_000,  ## 先頭10000件スキップが推奨
    ),
    DatasetSource(
        name="wikimedia/wikipedia",
        config_name="20231101.ja",
        split="train",
        take_examples=13500,
    ),
    DatasetSource(
        name="codeparrot/codeparrot-clean",
        split="train",
        take_examples=10_000,
    ),
]


SOURCES_JA45_EN45_CODE1: list[DatasetSource] = [
    DatasetSource(
        name="hotchpotch/fineweb-2-edu-japanese",
        config_name="small_tokens_cleaned",
        split="train",
        take_examples=36_000,
        skip_examples=10_000,  ## 先頭10000件スキップが推奨
    ),
    DatasetSource(
        name="wikimedia/wikipedia",
        config_name="20231101.ja",
        split="train",
        take_examples=9_000,
    ),
    DatasetSource(
        name="HuggingFaceFW/fineweb-edu",
        config_name="sample-10BT",
        split="train",
        take_examples=36_000,
    ),
    DatasetSource(
        name="wikimedia/wikipedia",
        config_name="20231101.en",
        split="train",
        take_examples=9_000,
    ),
    DatasetSource(
        name="codeparrot/codeparrot-clean",
        split="train",
        take_examples=10_000,
    ),
]


## 8:1:1
## chatGPTの概算では通常のtokenizerで1B Token程度
## byte換算で4B
SOURCES_JA8_EN1_CODE1_10: list[DatasetSource] = [
    DatasetSource(
        name="hotchpotch/fineweb-2-edu-japanese",
        config_name="small_tokens_cleaned",
        split="train",
        take_examples=680_000,
        skip_examples=10_000,  ## 先頭10000件スキップが推奨
    ),
    DatasetSource(
        name="wikimedia/wikipedia",
        config_name="20231101.ja",
        split="train",
        take_examples=120_000,
    ),
    DatasetSource(
        name="HuggingFaceFW/fineweb-edu",
        config_name="sample-10BT",
        split="train",
        take_examples=80_000,
    ),
    DatasetSource(
        name="wikimedia/wikipedia",
        config_name="20231101.en",
        split="train",
        take_examples=20_000,
    ),
    DatasetSource(
        name="codeparrot/codeparrot-clean",
        split="train",
        take_examples=100_000,
    ),
]


# 9:0:1
SOURCES_JA9_EN0_CODE1_10: list[DatasetSource] = [
    DatasetSource(
        name="hotchpotch/fineweb-2-edu-japanese",
        config_name="small_tokens_cleaned",
        split="train",
        take_examples=765_000,
        skip_examples=10_000,  ## 先頭10000件スキップが推奨
    ),
    DatasetSource(
        name="wikimedia/wikipedia",
        config_name="20231101.ja",
        split="train",
        take_examples=135_000,
    ),
    DatasetSource(
        name="codeparrot/codeparrot-clean",
        split="train",
        take_examples=100_000,
    ),
]


SOURCES_JA45_EN45_CODE1_10: list[DatasetSource] = [
    DatasetSource(
        name="hotchpotch/fineweb-2-edu-japanese",
        config_name="small_tokens_cleaned",
        split="train",
        take_examples=360_000,
        skip_examples=10_000,  ## 先頭10000件スキップが推奨
    ),
    DatasetSource(
        name="wikimedia/wikipedia",
        config_name="20231101.ja",
        split="train",
        take_examples=90_000,
    ),
    DatasetSource(
        name="HuggingFaceFW/fineweb-edu",
        config_name="sample-10BT",
        split="train",
        take_examples=360_000,
    ),
    DatasetSource(
        name="wikimedia/wikipedia",
        config_name="20231101.en",
        split="train",
        take_examples=90_000,
    ),
    DatasetSource(
        name="codeparrot/codeparrot-clean",
        split="train",
        take_examples=100_000,
    ),
    DatasetSource(
        name="if001/aozorabunko-clean-sin",
        split="train",
        take_examples=10_000,
    ),
]


SOURCES_JA8_EN1_CODE1_SMALL: list[DatasetSource] = [
    DatasetSource(
        name="hotchpotch/fineweb-2-edu-japanese",
        config_name="small_tokens_cleaned",
        split="train",
        take_examples=68_000,
        skip_examples=10_000,  ## 先頭10000件スキップが推奨
    ),
    DatasetSource(
        name="wikimedia/wikipedia",
        config_name="20231101.ja",
        split="train",
        take_examples=12_000,
    ),
    DatasetSource(
        name="HuggingFaceFW/fineweb-edu",
        config_name="sample-10BT",
        split="train",
        take_examples=8_000,
    ),
    DatasetSource(
        name="wikimedia/wikipedia",
        config_name="20231101.en",
        split="train",
        take_examples=2_000,
    ),
    DatasetSource(
        name="codeparrot/codeparrot-clean",
        split="train",
        take_examples=10_000,
    ),
    DatasetSource(
        name="if001/aozorabunko-clean-sin",
        split="train",
        take_examples=5_000,
    ),
]

## 8:1:1
## byte換算で12B
SOURCES_JA8_EN1_CODE1_20: list[DatasetSource] = [
    DatasetSource(
        name="hotchpotch/fineweb-2-edu-japanese",
        config_name="small_tokens_cleaned",
        split="train",
        take_examples=1_260_000,
        skip_examples=78_000,
    ),
    DatasetSource(
        name="wikimedia/wikipedia",
        config_name="20231101.ja",
        split="train",
        take_examples=240_000,
        skip_examples=12_000,
    ),
    DatasetSource(
        name="HuggingFaceFW/fineweb-edu",
        config_name="sample-10BT",
        split="train",
        take_examples=160_000,
        skip_examples=10_000,  ## 先頭10000件スキップが推奨
    ),
    DatasetSource(
        name="wikimedia/wikipedia",
        config_name="20231101.en",
        split="train",
        take_examples=40_000,
        skip_examples=2_000,
    ),
    DatasetSource(
        name="codeparrot/codeparrot-clean",
        split="train",
        take_examples=200_000,
        skip_examples=10_000,
    ),
    DatasetSource(
        name="if001/aozorabunko-clean-sin",
        split="train",
        take_examples=5_000,
        skip_examples=5_000,
    ),
]


## 8:1:1
## byte換算で12B
SOURCES_JA8_EN1_CODE1_40: list[DatasetSource] = [
    DatasetSource(
        name="hotchpotch/fineweb-2-edu-japanese",
        config_name="small_tokens_cleaned",
        split="train",
        take_examples=680_000 * 3,
        skip_examples=78_000,
    ),
    DatasetSource(
        name="wikimedia/wikipedia",
        config_name="20231101.ja",
        split="train",
        take_examples=120_000 * 3,
        skip_examples=12_000,
    ),
    DatasetSource(
        name="HuggingFaceFW/fineweb-edu",
        config_name="sample-10BT",
        split="train",
        take_examples=80_000 * 3,
        skip_examples=10_000,  ## 先頭10000件スキップが推奨
    ),
    DatasetSource(
        name="wikimedia/wikipedia",
        config_name="20231101.en",
        split="train",
        take_examples=20_000 * 3,
        skip_examples=2_000,
    ),
    DatasetSource(
        name="codeparrot/codeparrot-clean",
        split="train",
        take_examples=100_000 * 3,
        skip_examples=10_000,
    ),
    DatasetSource(
        name="if001/aozorabunko-clean-sin",
        split="train",
        take_examples=5_000,
        skip_examples=5_000,
    ),
]

from hnet.training import DatasetSource


## 8:1:1
SOURCES_JA8_EN1_CODE1: list[DatasetSource] = [
    DatasetSource(
        name="hotchpotch/fineweb-2-edu-japanese",
        config_name="small_tokens_cleaned",
        split="train",
        take_examples=680_000,
        skip_examples=10_000,
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
SOURCES_JA9_EN0_CODE1: list[DatasetSource] = [
    DatasetSource(
        name="hotchpotch/fineweb-2-edu-japanese",
        config_name="small_tokens_cleaned",
        split="train",
        take_examples=765_000,
        skip_examples=10_000,
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


SOURCES_JA45_EN45_CODE1: list[DatasetSource] = [
    DatasetSource(
        name="hotchpotch/fineweb-2-edu-japanese",
        config_name="small_tokens_cleaned",
        split="train",
        take_examples=360_000,
        skip_examples=10_000,
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
]

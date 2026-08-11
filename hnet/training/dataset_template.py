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

# JA only
SOURCES_JA: list[DatasetSource] = [
    DatasetSource(
        name="hotchpotch/fineweb-2-edu-japanese",
        config_name="small_tokens_cleaned",
        split="train",
        take_examples=5_000,
        skip_examples=10_000,  ## 先頭10000件スキップが推奨
    ),
    DatasetSource(
        name="wikimedia/wikipedia",
        config_name="20231101.ja",
        split="train",
        take_examples=5_000,
    ),
]
# CODE only
SOURCES_CODE: list[DatasetSource] = [
    DatasetSource(
        name="codeparrot/codeparrot-clean",
        split="train",
        take_examples=10_000,
    ),
]
# EN only
SOURCES_EN: list[DatasetSource] = [
    DatasetSource(
        name="HuggingFaceFW/fineweb-edu",
        config_name="sample-10BT",
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


SOURCES_JA8_EN1_CODE1_5: list[DatasetSource] = [
    DatasetSource(
        name="hotchpotch/fineweb-2-edu-japanese",
        config_name="small_tokens_cleaned",
        split="train",
        take_examples=68_000 * 5,
        skip_examples=10_000,  ## 先頭10000件スキップが推奨
    ),
    DatasetSource(
        name="wikimedia/wikipedia",
        config_name="20231101.ja",
        split="train",
        take_examples=12_000 * 5,
    ),
    DatasetSource(
        name="HuggingFaceFW/fineweb-edu",
        config_name="sample-10BT",
        split="train",
        take_examples=8_000 * 5,
    ),
    DatasetSource(
        name="wikimedia/wikipedia",
        config_name="20231101.en",
        split="train",
        take_examples=2_000 * 5,
    ),
    DatasetSource(
        name="codeparrot/codeparrot-clean",
        split="train",
        take_examples=10_000 * 5,
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
    # DatasetSource(
    #     name="if001/aozorabunko-clean-sin",
    #     split="train",
    #     take_examples=5_000,
    # ),
]


# Approximately 0.25B packed byte tokens with an observed byte mix near 8:1:1.
# Counts are calibrated from a 2026-07-29 packing of SOURCES_JA8_EN1_CODE1;
# source record sizes can change when upstream datasets are revised.
SOURCES_JA8_EN1_CODE1_SCREENING: list[DatasetSource] = [
    DatasetSource(
        name="hotchpotch/fineweb-2-edu-japanese",
        config_name="small_tokens_cleaned",
        split="train",
        take_examples=50_000,
        skip_examples=10_000,
    ),
    DatasetSource(
        name="wikimedia/wikipedia",
        config_name="20231101.ja",
        split="train",
        take_examples=6_000,
    ),
    DatasetSource(
        name="HuggingFaceFW/fineweb-edu",
        config_name="sample-10BT",
        split="train",
        take_examples=3_000,
    ),
    DatasetSource(
        name="wikimedia/wikipedia",
        config_name="20231101.en",
        split="train",
        take_examples=500,
    ),
    DatasetSource(
        name="codeparrot/codeparrot-clean",
        split="train",
        take_examples=2_567,
    ),
]


# Diversity control for SOURCES_JA8_EN1_CODE1_SCREENING. The English and code
# sources are unchanged, while approximately 60M Japanese bytes from standard
# Wikipedia are replaced by long-form Wikipedia and Aozora Bunko. Counts are
# calibrated from a 2026-07-31 streaming byte audit and should produce roughly
# the same 0.25B total packed bytes as the screening baseline.
SOURCES_JA8_EN1_CODE1_SCREENING_DIVERSITY: list[DatasetSource] = [
    DatasetSource(
        name="hotchpotch/fineweb-2-edu-japanese",
        config_name="small_tokens_cleaned",
        split="train",
        take_examples=50_000,
        skip_examples=10_000,
    ),
    DatasetSource(
        name="wikimedia/wikipedia",
        config_name="20231101.ja",
        split="train",
        take_examples=3_000,
    ),
    DatasetSource(
        name="if001/wikimedia_ja_long",
        split="train",
        take_examples=350,
    ),
    DatasetSource(
        name="if001/aozorabunko-clean-sin",
        split="train",
        take_examples=430,
        skip_examples=1_000,
    ),
    DatasetSource(
        name="HuggingFaceFW/fineweb-edu",
        config_name="sample-10BT",
        split="train",
        take_examples=3_000,
    ),
    DatasetSource(
        name="wikimedia/wikipedia",
        config_name="20231101.en",
        split="train",
        take_examples=500,
    ),
    DatasetSource(
        name="codeparrot/codeparrot-clean",
        split="train",
        take_examples=2_567,
    ),
]


# Fixed, non-overlapping holdout for SOURCES_JA8_EN1_CODE1_SCREENING.
# Expected packed size is approximately 12.5M byte tokens with the same mix.
SOURCES_JA8_EN1_CODE1_SCREENING_VALIDATION: list[DatasetSource] = [
    DatasetSource(
        name="hotchpotch/fineweb-2-edu-japanese",
        config_name="small_tokens_cleaned",
        split="train",
        take_examples=2_500,
        skip_examples=60_000,
    ),
    DatasetSource(
        name="wikimedia/wikipedia",
        config_name="20231101.ja",
        split="train",
        take_examples=300,
        skip_examples=6_000,
    ),
    DatasetSource(
        name="HuggingFaceFW/fineweb-edu",
        config_name="sample-10BT",
        split="train",
        take_examples=150,
        skip_examples=3_000,
    ),
    DatasetSource(
        name="wikimedia/wikipedia",
        config_name="20231101.en",
        split="train",
        take_examples=25,
        skip_examples=500,
    ),
    DatasetSource(
        name="codeparrot/codeparrot-clean",
        split="train",
        take_examples=128,
        skip_examples=2_567,
    ),
]

## 8:1:1
## byte換算で12B
## 7_346_735_500(7B)
SOURCES_JA8_EN1_CODE1_20: list[DatasetSource] = [
    DatasetSource(
        name="hotchpotch/fineweb-2-edu-japanese",
        config_name="small_tokens_cleaned",
        split="train",
        take_examples=1_260_000,
        skip_examples=78_000,
    ),
    DatasetSource(
        name="if001/wikimedia_ja_short",
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
        name="if001/wikimedia_en_short",
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
## byte換算で3B
SOURCES_JA8_EN1_CODE1_30: list[DatasetSource] = [
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
        take_examples=10_000,
    ),
]

## 8:1:1
## byte換算で
SOURCES_JA8_EN1_CODE1_50: list[DatasetSource] = [
    DatasetSource(
        name="hotchpotch/fineweb-2-edu-japanese",
        config_name="small_tokens_cleaned",
        split="train",
        take_examples=680_000 * 5,
        skip_examples=78_000,
    ),
    DatasetSource(
        name="wikimedia/wikipedia",
        config_name="20231101.ja",
        split="train",
        take_examples=120_000 * 5,
        skip_examples=12_000,
    ),
    DatasetSource(
        name="HuggingFaceFW/fineweb-edu",
        config_name="sample-10BT",
        split="train",
        take_examples=80_000 * 5,
        skip_examples=10_000,  ## 先頭10000件スキップが推奨
    ),
    DatasetSource(
        name="wikimedia/wikipedia",
        config_name="20231101.en",
        split="train",
        take_examples=20_000 * 5,
        skip_examples=2_000,
    ),
    DatasetSource(
        name="codeparrot/codeparrot-clean",
        split="train",
        take_examples=100_000 * 5,
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
## byte換算で
SOURCES_JA8_EN1_CODE1_100: list[DatasetSource] = [
    DatasetSource(
        name="hotchpotch/fineweb-2-edu-japanese",  # max: 120_000_000
        config_name="small_tokens_cleaned",
        split="train",
        take_examples=680_000 * 10,
        skip_examples=78_000,  ## 先頭10000件スキップが推奨
    ),
    DatasetSource(
        name="wikimedia/wikipedia",
        config_name="20231101.ja",
        split="train",
        take_examples=120_000 * 10,
        skip_examples=12_000,
    ),
    DatasetSource(
        name="HuggingFaceFW/fineweb-edu",
        config_name="sample-10BT",
        split="train",
        take_examples=80_000 * 10,
    ),
    DatasetSource(
        name="wikimedia/wikipedia",
        config_name="20231101.en",
        split="train",
        take_examples=20_000 * 10,
        skip_examples=2_000,
    ),
    DatasetSource(
        name="codeparrot/codeparrot-clean",
        split="train",
        take_examples=100_000 * 10,
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
## byte換算で60_830_092_012(60B)
SOURCES_JA8_EN1_CODE1_200: list[DatasetSource] = [
    DatasetSource(
        name="hotchpotch/fineweb-2-edu-japanese",  # max: 120_000_000
        config_name="small_tokens_cleaned",
        split="train",
        take_examples=680_000 * 20,
        skip_examples=78_000,  ## 先頭10000件スキップが推奨
    ),
    DatasetSource(
        name="if001/wikimedia_ja_short",
        split="train",
        take_examples=120_000 * 20,  # max: 1_380_000
        skip_examples=12_000,
    ),
    DatasetSource(
        name="HuggingFaceFW/fineweb-edu",
        config_name="sample-10BT",
        split="train",
        take_examples=80_000 * 20,
    ),
    DatasetSource(
        name="if001/wikimedia_en_short",
        split="train",
        take_examples=20_000 * 20,
        skip_examples=2_000,
    ),
    DatasetSource(
        name="codeparrot/codeparrot-clean",
        split="train",
        take_examples=100_000 * 20,
        skip_examples=10_000,
    ),
    DatasetSource(
        name="if001/aozorabunko-clean-sin",
        split="train",
        take_examples=1_000,
    ),
]


## long
SOURCES_JA8_EN1_CODE1_SMALL_LONG: list[DatasetSource] = [
    DatasetSource(
        name="hotchpotch/fineweb-2-edu-japanese",
        config_name="small_tokens_cleaned",
        split="train",
        take_examples=68_000,
        skip_examples=10_000,  ## 先頭10000件スキップが推奨
    ),
    DatasetSource(
        name="if001/wikimedia_ja_long",
        split="train",
        take_examples=4_000,
    ),
    DatasetSource(
        name="HuggingFaceFW/fineweb-edu",
        config_name="sample-10BT",
        split="train",
        take_examples=8_000,
    ),
    DatasetSource(
        name="if001/wikimedia_en_long",
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
        skip_examples=1_000,
        take_examples=10_000,
    ),
]

from hnet.training.dataset_template import (
    SOURCES_JA8_EN1_CODE1_SCREENING,
    SOURCES_JA8_EN1_CODE1_SCREENING_DIVERSITY,
    SOURCES_JA8_EN1_CODE1_SCREENING_VALIDATION,
)


def _source_key(source):
    return source.name, source.config_name, source.split


def test_screening_validation_is_after_training_sources() -> None:
    train_by_source = {
        _source_key(source): source for source in SOURCES_JA8_EN1_CODE1_SCREENING
    }
    validation_by_source = {
        _source_key(source): source
        for source in SOURCES_JA8_EN1_CODE1_SCREENING_VALIDATION
    }

    assert validation_by_source.keys() == train_by_source.keys()
    for key, train_source in train_by_source.items():
        validation_source = validation_by_source[key]
        assert train_source.take_examples > 0
        assert validation_source.take_examples > 0
        assert validation_source.skip_examples >= (
            train_source.skip_examples + train_source.take_examples
        )


def test_screening_diversity_keeps_fixed_sources_and_avoids_holdout() -> None:
    baseline = {
        _source_key(source): source for source in SOURCES_JA8_EN1_CODE1_SCREENING
    }
    diversity = {
        _source_key(source): source
        for source in SOURCES_JA8_EN1_CODE1_SCREENING_DIVERSITY
    }
    validation = {
        _source_key(source): source
        for source in SOURCES_JA8_EN1_CODE1_SCREENING_VALIDATION
    }

    fixed_keys = {
        ("hotchpotch/fineweb-2-edu-japanese", "small_tokens_cleaned", "train"),
        ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train"),
        ("wikimedia/wikipedia", "20231101.en", "train"),
        ("codeparrot/codeparrot-clean", None, "train"),
    }
    for key in fixed_keys:
        assert diversity[key] == baseline[key]

    assert ("if001/wikimedia_ja_long", None, "train") in diversity
    assert ("if001/aozorabunko-clean-sin", None, "train") in diversity

    for key in fixed_keys:
        if key not in validation:
            continue
        train_source = diversity[key]
        validation_source = validation[key]
        assert validation_source.skip_examples >= (
            train_source.skip_examples + train_source.take_examples
        )

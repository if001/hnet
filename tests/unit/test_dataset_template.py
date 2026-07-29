from hnet.training.dataset_template import (
    SOURCES_JA8_EN1_CODE1_SCREENING,
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

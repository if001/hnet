from hnet.training.trainer import crossed_interval


def test_crossed_interval_detects_threshold_crossing() -> None:
    assert not crossed_interval(0, 99, 100)
    assert crossed_interval(99, 100, 100)
    assert crossed_interval(199, 301, 100)


def test_crossed_interval_ignores_disabled_or_reverse_progress() -> None:
    assert not crossed_interval(0, 100, None)
    assert not crossed_interval(0, 100, 0)
    assert not crossed_interval(100, 99, 100)

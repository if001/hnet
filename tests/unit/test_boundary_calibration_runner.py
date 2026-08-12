from scripts.run_boundary_calibration import ratio_tag, run_name


def test_ratio_tag_and_run_name_are_stable() -> None:
    assert ratio_tag(0.03) == "0p03"
    assert ratio_tag(0.05) == "0p05"
    assert ratio_tag(0.08) == "0p08"
    assert (
        run_name("k1t1", 0.05, "1234567890")
        == "r5_cal_k1t1_comp3-3_rw0p05_utf8hard_s42_step55_1234567"
    )

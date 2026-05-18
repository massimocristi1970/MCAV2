from app.config.industry_config import (
    DIRECTOR_SCORE_PASS_THRESHOLD,
    INDUSTRY_THRESHOLDS,
    get_industry_thresholds,
    get_sector_risk,
)


def test_director_score_pass_threshold_is_applied_to_all_industries():
    assert DIRECTOR_SCORE_PASS_THRESHOLD == 68
    assert INDUSTRY_THRESHOLDS
    assert {
        thresholds["Directors Score"]
        for thresholds in INDUSTRY_THRESHOLDS.values()
    } == {DIRECTOR_SCORE_PASS_THRESHOLD}


def test_sector_risk_helper_matches_threshold_table():
    for industry, thresholds in INDUSTRY_THRESHOLDS.items():
        assert get_sector_risk(industry) == thresholds["Sector Risk"]


def test_unknown_industry_falls_back_to_other_thresholds():
    assert get_industry_thresholds("Not A Real Industry") == INDUSTRY_THRESHOLDS["Other"]
    assert get_sector_risk("Not A Real Industry") == INDUSTRY_THRESHOLDS["Other"]["Sector Risk"]

from datetime import datetime
import xml.etree.ElementTree as ET

import pytest

from app.main import combine_mca_and_tu_decisions
from src.tu_scorecard.feature_extractor import (
    _extract_account_history_features,
    _extract_search_recency_features,
    extract_features_from_xml_bytes,
)
from src.tu_scorecard.scorecard_rules import score_tu_features


@pytest.mark.unit
def test_arrears_count_distinct_months_not_account_months():
    applicant = ET.fromstring(
        """
        <applicant><accs>
          <acc><acchistory><ah m="2026-07" pay="1"/></acchistory></acc>
          <acc><acchistory><ah m="2026-07" pay="2"/></acchistory></acc>
        </accs></applicant>
        """
    )

    features = _extract_account_history_features(
        applicant, datetime(2026, 7, 28)
    )

    assert features["account_history_records_total"] == 2
    assert features["missed_months_3m"] == 1
    assert features["missed_months_6m"] == 1
    assert features["missed_months_12m"] == 1
    assert features["worst_ah_pay_12m"] == 2


@pytest.mark.unit
def test_nested_account_history_flag_requires_decoded_history():
    without_history = b"""
    <Envelope><Body><Search07aResponse><SearchResult>
      <jobdetails><searchdate>2026-07-28T10:00:00</searchdate></jobdetails>
      <creditreport><applicant><accs><acc/></accs></applicant></creditreport>
    </SearchResult></Search07aResponse></Body></Envelope>
    """
    with_history = without_history.replace(
        b"<acc/>", b'<acc><acchistory><ah m="2026-07" pay="0"/></acchistory></acc>'
    )

    assert extract_features_from_xml_bytes(
        without_history, "without-history"
    ).features["account_history_extracted"] is False
    assert extract_features_from_xml_bytes(
        with_history, "with-history"
    ).features["account_history_extracted"] is True


@pytest.mark.unit
def test_future_searches_are_excluded_from_recency_counts():
    applicant = ET.fromstring(
        """
        <applicant><searches>
          <search><searchdate>2026-07-20</searchdate></search>
          <search><searchdate>2026-08-01</searchdate></search>
        </searches></applicant>
        """
    )

    features = _extract_search_recency_features(
        applicant, datetime(2026, 7, 28)
    )

    assert features == {"searches_30d": 1, "searches_90d": 1}


@pytest.mark.unit
def test_future_opened_accounts_are_excluded_from_fixed_width_window():
    payload = (
        '<rows><row><field name="Search07aResponse">'
        '2026-07-28T10:00:00GBP1002026-07-01GBP2002026-08-01'
        '</field></row></rows>'
    ).encode()

    features = extract_features_from_xml_bytes(payload, "future-account").features

    assert features["accounts_opened_6m_total"] == 1


@pytest.mark.unit
def test_incomplete_tu_data_is_capped_at_refer():
    features = {
        "accounts_total": 40,
        "accounts_settled_total": 30,
        "bureau_score": 700,
        "bureau_score_extracted": False,
        "account_summary_extracted": True,
        "account_history_extracted": True,
        "search_recency_extracted": True,
    }

    result = score_tu_features(features)

    assert result.score == 59
    assert result.decision == "REFER"
    assert result.reasons[0] == "Incomplete TU data: missing bureau score"


@pytest.mark.unit
def test_hard_decline_takes_precedence_over_incomplete_data():
    result = score_tu_features(
        {
            "active_judgments_total": 1,
            "bureau_score_extracted": False,
            "account_summary_extracted": False,
            "account_history_extracted": False,
            "search_recency_extracted": False,
        }
    )

    assert result.score == 0
    assert result.decision == "DECLINE"


@pytest.mark.unit
@pytest.mark.parametrize("mca_decision", ["APPROVE", "CONDITIONAL_APPROVE", "REFER"])
def test_tu_decline_remains_final_decline(mca_decision):
    assert combine_mca_and_tu_decisions(mca_decision, "DECLINE") == "DECLINE"


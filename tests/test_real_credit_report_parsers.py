from pathlib import Path
from dataclasses import asdict, is_dataclass

import pytest

from app.main import (
    _bureau_band_from_pdf_text,
    _explicit_ccj_present,
    _extract_text_from_pdf_bytes,
    _parse_business_bureau_signals,
    build_report_information,
)
from src.tu_scorecard.feature_extractor import extract_features_from_xml_bytes
from src.tu_scorecard.scorecard_rules import score_tu_features


TU_XML = Path(
    r"C:\Users\Massimo Cristi\OneDrive - Savvy Loan Products Ltd\Merchant Cash Advance (MCA)\Applications\UNDERWRITING\firstunicorn ltd\101897.xml"
)
BUSINESS_PDF = Path(
    r"C:\Users\Massimo Cristi\OneDrive - Savvy Loan Products Ltd\Merchant Cash Advance (MCA)\Applications\UNDERWRITING\firstunicorn ltd\FIRSTUNICORN_LTD-capital_report-2026-05-11.pdf"
)
IRVING_TU_XML = Path(
    r"D:\OneDrive - Savvy Loan Products Ltd\Merchant Cash Advance (MCA)\Applications\DECLINED APPLICATIONS\E & R IRVING FISHMONGERS LTD\101901.xml"
)


@pytest.mark.unit
def test_real_tu_search07a_wrapper_extracts_director_score_features():
    if not TU_XML.exists():
        pytest.skip("Real TU XML fixture is not available on this machine")

    features_obj = extract_features_from_xml_bytes(TU_XML.read_bytes(), app_id="101897")
    payload = asdict(features_obj) if is_dataclass(features_obj) else dict(features_obj)
    features = payload.get("features", payload)
    result = score_tu_features(features)

    assert features["tu_parser"] == "fixed_width_search07a_response"
    assert features["bureau_score"] == 857
    assert features["accounts_total"] == 17
    assert features["accounts_active_total"] == 17
    assert features["accounts_settled_total"] == 6
    assert features["accounts_opened_6m_total"] == 6
    assert result.score == 60
    assert result.decision == "APPROVE"


@pytest.mark.unit
def test_real_tu_active_iva_is_hard_decline_when_available():
    if not IRVING_TU_XML.exists():
        pytest.skip("Real TU XML fixture is not available on this machine")

    features_obj = extract_features_from_xml_bytes(IRVING_TU_XML.read_bytes(), app_id="101901")
    features = asdict(features_obj)["features"]
    result = score_tu_features(features)

    assert features["tu_parser"] == "fixed_width_search07a_response"
    assert features["current_bai_record"] is True
    assert features["active_iva_or_admin_order"] is True
    assert features["bureau_score"] == 0
    assert result.score == 0
    assert result.decision == "DECLINE"
    assert any("Active IVA" in reason for reason in result.reasons)


@pytest.mark.unit
def test_active_iva_is_used_regardless_of_date():
    features = {
        "current_bai_record": True,
        "active_iva_or_admin_order": True,
        "public_record_reasons": ["Active IVA/administration order recorded from 2019-01-01"],
        "accounts_total": 40,
        "accounts_settled_total": 30,
        "bureau_score": 700,
    }

    result = score_tu_features(features)

    assert result.score == 0
    assert result.decision == "DECLINE"
    assert result.reasons == ["Active IVA/administration order recorded from 2019-01-01"]


@pytest.mark.unit
def test_discharged_iva_is_used_inside_36_month_window_only():
    recent = {
        "iva_or_admin_order_36m": True,
        "public_record_reasons": ["IVA/administration order recorded within 36 months (2025-01-01)"],
        "accounts_total": 40,
        "accounts_settled_total": 30,
        "bureau_score": 700,
    }
    old = {
        "iva_or_admin_order_36m": False,
        "accounts_total": 40,
        "accounts_settled_total": 30,
        "bureau_score": 700,
    }

    recent_result = score_tu_features(recent)
    old_result = score_tu_features(old)

    assert recent_result.score == 0
    assert recent_result.decision == "DECLINE"
    assert old_result.decision == "APPROVE"


@pytest.mark.unit
def test_real_capital_business_pdf_extracts_bureau_signals():
    if not BUSINESS_PDF.exists():
        pytest.skip("Real business credit PDF fixture is not available on this machine")

    text, backend, error = _extract_text_from_pdf_bytes(BUSINESS_PDF.read_bytes())
    signals = _parse_business_bureau_signals(text)
    band, reasons = _bureau_band_from_pdf_text(text)
    info = build_report_information(text)

    assert backend != "none", error
    assert _explicit_ccj_present(text) is False
    assert signals["credit_score"] is None
    assert signals["credit_score_suppressed"] is True
    assert signals["credit_limit"] == 0
    assert signals["max_recommended_credit"] == 0
    assert signals["enquiries_3m"] == 3
    assert signals["negative_impact_count"] == 4
    assert signals["no_ccj_recorded"] is True
    assert band == "C (High Risk)"
    assert "Credit limit and max recommended credit are £0" in reasons

    public_record = "\n".join(info["Public Record"])
    assert "Company name: FIRSTUNICORN LTD" in public_record
    assert "Company number: 15064523" in public_record
    assert "Company Status: Active" in public_record


@pytest.mark.unit
def test_capital_credit_score_range_is_usable_bureau_score():
    text = """
    Credit information Needs attention
    Credit score
    16 - 25
    E
    Credit limit Max. recommended credit
    £500
    Company searches
    In the last 12 months
    11
    7 enquiries in last 3 months
    Credit risk factors Needs attention
    Total factors
    10
    Negative impact: 2
    Neutral impact: 8
    No CCJs recorded
    """

    signals = _parse_business_bureau_signals(text)
    band, reasons = _bureau_band_from_pdf_text(text)
    info = build_report_information(text)

    assert signals["credit_score"] == 25
    assert signals["credit_score_min"] == 16
    assert signals["credit_score_max"] == 25
    assert signals["credit_score_range"] == "16-25"
    assert signals["credit_score_suppressed"] is False
    assert band == "D (Very High Risk)"
    assert "Credit score: 16-25" in info["Credit information"]

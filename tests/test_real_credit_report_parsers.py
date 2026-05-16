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

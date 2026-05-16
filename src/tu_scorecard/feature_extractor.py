from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Any, Dict, Optional
from xml.etree.ElementTree import ParseError

from .xml_utils import (
    find_first_by_local,
    find_all_by_local,
    get_text,
    to_int,
    to_float,
    strip_ns,
)

# Configure module logger
logger = logging.getLogger(__name__)



@dataclass
class TUFeatures:
    app_id: str
    anchor_date: Optional[datetime]
    features: Dict[str, Any]


def _parse_dt(s: str) -> Optional[datetime]:
    if not s:
        return None
    # Examples seen: 2023-03-01T21:50:41
    try:
        return datetime.fromisoformat(s.replace("Z", ""))
    except Exception:
        return None

_XML_10_ILLEGAL = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F]"
)

def _sanitize_xml_bytes(xml_bytes: bytes) -> bytes:
    """
    Remove characters that are illegal in XML 1.0.
    Strategy:
    - decode as utf-8 (with replacement so we don't crash)
    - remove control chars not allowed in XML 1.0
    - re-encode to utf-8
    """
    text = xml_bytes.decode("utf-8", errors="replace")
    text = _XML_10_ILLEGAL.sub("", text)
    return text.encode("utf-8")



def _month_key(dt: datetime) -> str:
    return f"{dt.year:04d}-{dt.month:02d}"


def _extract_fixed_width_search07a_response(root: ET.Element, app_id: str) -> Optional[TUFeatures]:
    """
    Handle TU exports shaped as:
        <rows><row><field name="Search07aResponse">...</field></row></rows>

    The field payload is a fixed-width/plain-text Search07a response rather than
    nested XML. We extract only stable non-PII summary signals for the scorecard.
    """
    if strip_ns(root.tag).lower() != "rows":
        return None

    field = None
    for node in root.iter():
        if strip_ns(node.tag).lower() == "field" and node.attrib.get("name") == "Search07aResponse":
            field = node
            break

    if field is None or not (field.text or "").strip():
        return None

    payload = (field.text or "").strip()
    anchor_date = None
    first_dt = re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", payload)
    if first_dt:
        anchor_date = _parse_dt(first_dt.group(0))

    # Final score footer in this export appears as F###.... near the end.
    bureau_score = 0
    footer_score = re.search(r"F(\d{3})(?=\d{4}[A-Z]\d{4}\.\d{2}\s*$)", payload)
    if footer_score:
        bureau_score = to_int(footer_score.group(1))
    else:
        score_candidates = [int(m.group(1)) for m in re.finditer(r"F(\d{3})", payload) if 300 <= int(m.group(1)) <= 999]
        if score_candidates:
            bureau_score = score_candidates[-1]

    summary_counts = re.search(r"(\d{2})(\d{2})(\d{2})UU", payload)
    summary_accounts_total = None
    summary_accounts_active = None
    summary_accounts_settled = None
    if summary_counts:
        summary_accounts_total = to_int(summary_counts.group(1))
        summary_accounts_active = to_int(summary_counts.group(2))
        summary_accounts_settled = to_int(summary_counts.group(3))

    account_records = []
    for match in re.finditer(r"GBP(\d*)(?=\d{4}-\d{2}-\d{2})", payload):
        amount_text = match.group(1)
        balance = to_float(amount_text) if amount_text != "" else 0.0
        opened_match = re.match(r"(\d{4}-\d{2}-\d{2})", payload[match.end():])
        opened_date = _parse_dt(opened_match.group(1)) if opened_match else None
        account_records.append({"balance": balance, "opened_date": opened_date})

    accounts_total = summary_accounts_total if summary_accounts_total is not None else len(account_records)
    accounts_active = summary_accounts_active if summary_accounts_active is not None else sum(1 for record in account_records if record["balance"] > 0)
    accounts_settled = summary_accounts_settled if summary_accounts_settled is not None else sum(1 for record in account_records if record["balance"] <= 0)
    opened_6m = 0
    if anchor_date:
        cutoff_6m = anchor_date - relativedelta(months=6)
        opened_6m = sum(
            1 for record in account_records
            if record["opened_date"] is not None and record["opened_date"] >= cutoff_6m
        )

    feats: Dict[str, Any] = {
        "searches_3m_total": 0,
        "searches_12m_total": 0,
        "homecredit_searches_3m_total": 0,
        "ccj_active_total": 0,
        "ccj_satisfied_total": 0,
        "accounts_total": accounts_total,
        "accounts_active_total": accounts_active,
        "accounts_settled_total": accounts_settled,
        "accounts_opened_6m_total": opened_6m,
        "worst_pay_status_12m": 0,
        "worst_pay_status_36m": 0,
        "delinq_12m_total": 0,
        "delinq_36m_total": 0,
        "defaults_12m_total": 0,
        "defaults_36m_total": 0,
        "bureau_score": bureau_score,
        "missed_months_3m": 0,
        "missed_months_6m": 0,
        "missed_months_12m": 0,
        "worst_ah_pay_12m": 0,
        "sum_balances": 0.0,
        "sum_limits": 0.0,
        "utilisation_pct": 0.0,
        "searches_30d": 0,
        "searches_90d": 0,
        "tu_parser": "fixed_width_search07a_response",
    }

    return TUFeatures(app_id=app_id, anchor_date=anchor_date, features=feats)


class XMLParseError(Exception):
    """Raised when XML parsing fails even after sanitization."""
    pass


def extract_features_from_xml_bytes(xml_bytes: bytes, app_id: str) -> TUFeatures:
    """
    Extract stable bureau features from TU Search07a XML.
    Avoid PII: do NOT export names, full addresses, DOB, etc.
    
    Args:
        xml_bytes: Raw XML content as bytes
        app_id: Application identifier for logging/tracking
        
    Returns:
        TUFeatures object with extracted features
        
    Raises:
        XMLParseError: If XML cannot be parsed even after sanitization
    """
    root = None
    parse_error_msg = None
    
    # First attempt: parse raw XML
    try:
        root = ET.fromstring(xml_bytes)
        logger.debug(f"[{app_id}] XML parsed successfully on first attempt")
    except ParseError as e:
        parse_error_msg = str(e)
        logger.debug(f"[{app_id}] Initial XML parse failed: {parse_error_msg}, attempting sanitization")
        
        # Second attempt: sanitize and retry
        try:
            xml_bytes2 = _sanitize_xml_bytes(xml_bytes)
            root = ET.fromstring(xml_bytes2)
            logger.debug(f"[{app_id}] XML parsed successfully after sanitization")
        except ParseError as e2:
            error_msg = f"XML parsing failed for app_id={app_id}. Original error: {parse_error_msg}. After sanitization: {e2}"
            logger.error(error_msg)
            raise XMLParseError(error_msg) from e2
    
    if root is None:
        raise XMLParseError(f"Failed to parse XML for app_id={app_id}: root element is None")

    fixed_width_features = _extract_fixed_width_search07a_response(root, app_id)
    if fixed_width_features is not None:
        return fixed_width_features

    # Anchor date = jobdetails/searchdate (date the bureau search was run)
    anchor_text = get_text(
        find_first_by_local(
            root,
            "Body/Search07aResponse/SearchResult/jobdetails/searchdate",
        )
    )
    anchor_date = _parse_dt(anchor_text)

    # Core node shortcuts
    applicant = find_first_by_local(
        root,
        "Body/Search07aResponse/SearchResult/creditreport/applicant",
    )

    feats: Dict[str, Any] = {}

    # ---------------------------
    # Summary block (high value)
    # ---------------------------
    summary = None if applicant is None else next((c for c in list(applicant) if strip_ns(c.tag) == "summary"), None)

    def sum_path(local_name: str) -> Optional[ET.Element]:
        if summary is None:
            return None
        for c in list(summary):
            if strip_ns(c.tag) == local_name:
                return c
        return None

    # Summary/searches
    s_searches = sum_path("searches")
    if s_searches is not None:
        feats["searches_3m_total"] = to_int(get_text(find_first_by_local(s_searches, "totalsearches3months")))
        feats["searches_12m_total"] = to_int(get_text(find_first_by_local(s_searches, "totalsearches12months")))
        feats["homecredit_searches_3m_total"] = to_int(get_text(find_first_by_local(s_searches, "totalhomecreditsearches3months")))
    else:
        feats["searches_3m_total"] = 0
        feats["searches_12m_total"] = 0
        feats["homecredit_searches_3m_total"] = 0

    # Summary/judgments
    s_judgments = sum_path("judgments")
    if s_judgments is not None:
        feats["ccj_active_total"] = to_int(get_text(find_first_by_local(s_judgments, "totalactive")))
        feats["ccj_satisfied_total"] = to_int(get_text(find_first_by_local(s_judgments, "totalsatisfied")))
    else:
        feats["ccj_active_total"] = 0
        feats["ccj_satisfied_total"] = 0

    # Summary/share (accounts + worst status + delinq/default counts)
    s_share = sum_path("share")
    if s_share is not None:
        feats["accounts_total"] = to_int(get_text(find_first_by_local(s_share, "totalaccounts")))
        feats["accounts_active_total"] = to_int(get_text(find_first_by_local(s_share, "totalactiveaccs")))
        feats["accounts_settled_total"] = to_int(get_text(find_first_by_local(s_share, "totalsettledaccs")))
        feats["accounts_opened_6m_total"] = to_int(get_text(find_first_by_local(s_share, "totalopened6months")))
        feats["worst_pay_status_12m"] = to_int(get_text(find_first_by_local(s_share, "worsepaystatus12months")))
        feats["worst_pay_status_36m"] = to_int(get_text(find_first_by_local(s_share, "worsepaystatus36months")))
        feats["delinq_12m_total"] = to_int(get_text(find_first_by_local(s_share, "totaldelinqs12months")))
        feats["delinq_36m_total"] = to_int(get_text(find_first_by_local(s_share, "totaldelinqs36months")))
        feats["defaults_12m_total"] = to_int(get_text(find_first_by_local(s_share, "totaldefaults12months")))
        feats["defaults_36m_total"] = to_int(get_text(find_first_by_local(s_share, "totaldefaults36months")))
    else:
        feats.update(
            {
                "accounts_total": 0,
                "accounts_active_total": 0,
                "accounts_settled_total": 0,
                "accounts_opened_6m_total": 0,
                "worst_pay_status_12m": 0,
                "worst_pay_status_36m": 0,
                "delinq_12m_total": 0,
                "delinq_36m_total": 0,
                "defaults_12m_total": 0,
                "defaults_36m_total": 0,
            }
        )

    # ---------------------------
    # Bureau score (if present)
    # ---------------------------
    creditscores = None if applicant is None else next((c for c in list(applicant) if strip_ns(c.tag) == "creditscores"), None)
    bureau_score = None
    if creditscores is not None:
        # Take the first <creditscore><score> we find
        for cs in list(creditscores):
            if strip_ns(cs.tag) != "creditscore":
                continue
            score_el = find_first_by_local(cs, "score")
            s = get_text(score_el)
            if s:
                bureau_score = to_int(s, default=None)  # type: ignore[arg-type]
                break
    feats["bureau_score"] = bureau_score if bureau_score is not None else 0

    # ---------------------------
    # Account-history derived features
    # ---------------------------
    feats.update(_extract_account_history_features(applicant, anchor_date))

    # ---------------------------
    # Searches derived recency (if anchor_date present)
    # ---------------------------
    feats.update(_extract_search_recency_features(applicant, anchor_date))

    return TUFeatures(app_id=app_id, anchor_date=anchor_date, features=feats)


def _extract_search_recency_features(applicant: Optional[ET.Element], anchor_date: Optional[datetime]) -> Dict[str, Any]:
    out = {
        "searches_30d": 0,
        "searches_90d": 0,
    }
    if applicant is None or anchor_date is None:
        return out

    searches = next((c for c in list(applicant) if strip_ns(c.tag) == "searches"), None)
    if searches is None:
        return out

    cutoff_30 = anchor_date - relativedelta(days=30)
    cutoff_90 = anchor_date - relativedelta(days=90)

    for s in list(searches):
        if strip_ns(s.tag) != "search":
            continue
        d_text = get_text(find_first_by_local(s, "searchdate"))
        d = _parse_dt(d_text)
        if not d:
            continue
        if d >= cutoff_30:
            out["searches_30d"] += 1
        if d >= cutoff_90:
            out["searches_90d"] += 1

    return out


def _extract_account_history_features(applicant: Optional[ET.Element], anchor_date: Optional[datetime]) -> Dict[str, Any]:
    """
    Parse account <acchistory><ah m="YYYY-MM" pay="0|1|2|3|..."> to compute:
    - missed months last 3/6/12m
    - worst pay last 12m
    - simple utilisation (where credit limits exist)
    """
    out: Dict[str, Any] = {
        "missed_months_3m": 0,
        "missed_months_6m": 0,
        "missed_months_12m": 0,
        "worst_ah_pay_12m": 0,
        "sum_balances": 0.0,
        "sum_limits": 0.0,
        "utilisation_pct": 0.0,
    }
    if applicant is None or anchor_date is None:
        return out

    accs = next((c for c in list(applicant) if strip_ns(c.tag) == "accs"), None)
    if accs is None:
        return out

    # define month windows relative to anchor_date
    months_3 = {_month_key(anchor_date - relativedelta(months=i)) for i in range(0, 3)}
    months_6 = {_month_key(anchor_date - relativedelta(months=i)) for i in range(0, 6)}
    months_12 = {_month_key(anchor_date - relativedelta(months=i)) for i in range(0, 12)}

    worst_12m = 0

    for acc in list(accs):
        if strip_ns(acc.tag) != "acc":
            continue

        accdetails = next((c for c in list(acc) if strip_ns(c.tag) == "accdetails"), None)
        if accdetails is not None:
            bal = to_float(get_text(find_first_by_local(accdetails, "balance")))
            out["sum_balances"] += max(bal, 0.0)

            # Some feeds include limit/creditlimit. If absent, skip.
            limit_el = find_first_by_local(accdetails, "creditlimit")
            if limit_el is None:
                limit_el = find_first_by_local(accdetails, "limit")
            lim = to_float(get_text(limit_el))
            if lim > 0:
                out["sum_limits"] += lim

        acchistory = next((c for c in list(acc) if strip_ns(c.tag) == "acchistory"), None)
        if acchistory is None:
            continue

        for ah in list(acchistory):
            if strip_ns(ah.tag) != "ah":
                continue
            m = ah.attrib.get("m", "")
            pay_raw = ah.attrib.get("pay", "")
            if not m or pay_raw == "":
                continue

            # pay is often numeric string. Treat "0" as OK, >0 as missed/arrears
            pay = 0
            try:
                pay = int(pay_raw)
            except Exception:
                pay = 0

            if m in months_12:
                worst_12m = max(worst_12m, pay)
                if pay > 0:
                    out["missed_months_12m"] += 1
                    if m in months_6:
                        out["missed_months_6m"] += 1
                        if m in months_3:
                            out["missed_months_3m"] += 1

    out["worst_ah_pay_12m"] = worst_12m
    out["utilisation_pct"] = (out["sum_balances"] / out["sum_limits"] * 100.0) if out["sum_limits"] > 0 else 0.0
    return out

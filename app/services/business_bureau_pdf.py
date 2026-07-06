from __future__ import annotations

import re
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any


MIN_EXTRACTED_TEXT_CHARS = 40
POUND = "\u00a3"


@dataclass(frozen=True)
class BusinessBureauParseResult:
    text: str = ""
    backend: str = "none"
    error: str = ""
    parse_status: str = "missing"
    business_ccj: bool | None = None
    bureau_band: str | None = None
    bureau_band_reasons: list[str] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)
    report_information: dict[str, list[str]] = field(default_factory=dict)


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> tuple[str, str, str]:
    """
    Returns: (text, backend_used, error_msg).
    Tries multiple backends and reports why extraction failed.
    """
    if not pdf_bytes or not pdf_bytes[:4] == b"%PDF":
        return "", "none", "Not a valid PDF header (expected %PDF)."

    errors = []

    try:
        import pdfplumber

        parts = []
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                parts.append(page.extract_text() or "")
        text = "\n".join(parts).strip()
        if text:
            return text, "pdfplumber", ""
        errors.append("pdfplumber: extracted empty text")
    except Exception as exc:
        errors.append(f"pdfplumber: {repr(exc)}")

    try:
        import fitz

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        parts = []
        for page in doc:
            parts.append(page.get_text("text") or "")
        text = "\n".join(parts).strip()
        if text:
            return text, "pymupdf", ""
        errors.append("pymupdf: extracted empty text")
    except Exception as exc:
        errors.append(f"pymupdf: {repr(exc)}")

    try:
        import PyPDF2

        reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
        parts = [(page.extract_text() or "") for page in reader.pages]
        text = "\n".join(parts).strip()
        if text:
            return text, "pypdf2", ""
        errors.append("pypdf2: extracted empty text")
    except Exception as exc:
        errors.append(f"pypdf2: {repr(exc)}")

    return "", "none", " | ".join(errors)


def parse_business_bureau_pdf(pdf_bytes: bytes | None) -> BusinessBureauParseResult:
    if not pdf_bytes:
        return BusinessBureauParseResult()

    text, backend, error = extract_text_from_pdf_bytes(pdf_bytes)
    if backend == "none" or len((text or "").strip()) < MIN_EXTRACTED_TEXT_CHARS:
        return BusinessBureauParseResult(
            text=text or "",
            backend=backend,
            error=error or "PDF text extraction returned too little usable text.",
            parse_status="failed",
        )

    signals = parse_business_bureau_signals(text)
    band, reasons = bureau_band_from_pdf_text(text)
    return BusinessBureauParseResult(
        text=text,
        backend=backend,
        error=error,
        parse_status="parsed",
        business_ccj=explicit_ccj_present(text),
        bureau_band=band,
        bureau_band_reasons=reasons,
        signals=signals,
        report_information=build_report_information(text),
    )


def explicit_ccj_present(pdf_text: str) -> bool:
    """
    Detect explicit business CCJ evidence.
    Negative statements win. Positive detection is scoped to legal/public-record
    text where possible to avoid using unrelated dates and amounts.
    """
    t = norm_pdf_text(pdf_text or "").lower()

    negative_patterns = [
        r"\bno\s+county\s+court\s+judg(e)?ment(s)?\b",
        r"\bno\s+ccj(s)?\b",
        r"\bccj(s)?\s*:\s*(none|no|0)\b",
        r"\bnone\s+recorded\b[\s\S]{0,80}\bccj(s)?\b",
        r"\bno\s+ccj(s)?\s+recorded\b",
    ]
    for pattern in negative_patterns:
        if re.search(pattern, t, re.IGNORECASE):
            return False

    section = _section_text(t, ("legal notices", "public record"), ("charges", "credit information", "payment performance"))
    ccj_scope = section or t

    positive_patterns = [
        r"county\s+court\s+judg(e)?ment\s+registered",
        r"county\s+court\s+judg(e)?ments\s+registered",
        r"county\s+court\s+judg(e)?ment\s+has\s+been\s+registered",
        r"\bccj\s+registered\b",
        r"\bat\s+least\s+one\s+county\s+court\s+judg(e)?ment\b",
    ]
    for pattern in positive_patterns:
        if re.search(pattern, ccj_scope, re.IGNORECASE):
            return True

    if re.search(r"county\s+court\s+judg(e)?ments?\b", ccj_scope):
        has_money = re.search(rf"(?:{re.escape(POUND)}|gbp)\s*\d", ccj_scope, re.IGNORECASE)
        has_date = re.search(r"\b\d{1,2}\s+[a-z]{3,9}\s+\d{4}\b", ccj_scope)
        if has_money and has_date:
            return True

    return False


def norm_pdf_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\ufb01", "fi").replace("\ufb02", "fl")
    text = text.replace("\ufffd", "")
    text = text.replace("\x00", "fi")
    text = text.replace("\u00c5\u0081", POUND)
    text = text.replace("\u00c2\u00a3", POUND)
    return text


def _re_first(pattern: str, text: str, flags=re.IGNORECASE) -> str | None:
    match = re.search(pattern, text, flags)
    return match.group(1).strip() if match else None


def _money_clean(value: str | None) -> str | None:
    if not value:
        return None
    value = value.replace(",", "").strip()
    if value.startswith(POUND):
        return POUND + value[1:].strip()
    return value


def _money_to_int(value: str | None) -> int | None:
    if not value:
        return None
    digits = re.sub(r"[^\d]", "", str(value))
    return int(digits) if digits else None


def _section_text(text: str, starts: tuple[str, ...], ends: tuple[str, ...]) -> str:
    lowered = text.lower()
    start_positions = [lowered.find(start) for start in starts if lowered.find(start) >= 0]
    if not start_positions:
        return ""
    start = min(start_positions)
    end = len(text)
    for marker in ends:
        pos = lowered.find(marker, start + 1)
        if pos >= 0:
            end = min(end, pos)
    return text[start:end]


def parse_business_bureau_signals(full_text: str) -> dict[str, object]:
    text = norm_pdf_text(full_text or "")
    lowered = text.lower()
    money_pattern = rf"({re.escape(POUND)}\s*\d[\d,]*|GBP\s*\d[\d,]*)"

    credit_score_range = None
    credit_score_min = None
    credit_score_max = None
    match_range = re.search(
        r"\bcredit\s+score\b[\s\S]{0,120}?(\d{1,3})\s*[-\u2013\u2014]\s*(\d{1,3})\b",
        text,
        re.IGNORECASE,
    )
    if match_range:
        credit_score_min = int(match_range.group(1))
        credit_score_max = int(match_range.group(2))
        credit_score_range = f"{credit_score_min}-{credit_score_max}"

    credit_score = _re_first(r"\bcredit\s+score\b\s*[:\-]?\s*(?:\r?\n\s*)?(\d{1,3})\b", text)
    credit_limit = _re_first(rf"\bcredit\s+limit\b[\s\S]{{0,100}}?{money_pattern}", text)
    max_credit = _re_first(rf"\bmax\.?\s*recommended\s*credit\b[\s\S]{{0,120}}?{money_pattern}", text)

    searches_section = _section_text(text, ("company searches",), ("credit risk factors", "legal notices", "public record"))
    searches_12m = _re_first(r"\bin\s+the\s+last\s+12\s+months\s*(?:\r?\n\s*)?(\d+)\b", searches_section or text)
    enquiries_3m = _re_first(r"\b(\d+)\s+enquiries\s+in\s+(?:the\s+)?last\s+3\s+months\b", searches_section or text)

    negative_impact = _re_first(r"\bnegative\s+impact\s*:\s*(\d+)\b", text)
    neutral_impact = _re_first(r"\bneutral\s+impact\s*:\s*(\d+)\b", text)
    total_factors = _re_first(r"\btotal\s+factors\s*(?:\r?\n\s*)?(\d+)\b", text)

    no_ccj = bool(re.search(r"\bno\s+(?:county\s+court\s+judg(e)?ments?|ccj(s)?)\s+recorded\b", lowered))
    no_legal_notices = "no legal notices registered" in lowered
    no_charges = "no registered mortgages or charges" in lowered or "no mortgages or charges" in lowered

    return {
        "credit_score": credit_score_max if credit_score_range else (int(credit_score) if credit_score else None),
        "credit_score_min": credit_score_min,
        "credit_score_max": credit_score_max,
        "credit_score_range": credit_score_range,
        "credit_score_suppressed": "risk score suppressed" in lowered or bool(re.search(r"\bcredit score\s*-\s*risk score suppressed\b", lowered)),
        "credit_limit": _money_to_int(credit_limit),
        "max_recommended_credit": _money_to_int(max_credit),
        "company_searches_12m": int(searches_12m) if searches_12m else None,
        "enquiries_3m": int(enquiries_3m) if enquiries_3m else None,
        "negative_impact_count": int(negative_impact) if negative_impact else 0,
        "neutral_impact_count": int(neutral_impact) if neutral_impact else None,
        "total_factor_count": int(total_factors) if total_factors else None,
        "needs_attention": "needs attention" in lowered,
        "no_ccj_recorded": no_ccj,
        "no_legal_notices_registered": no_legal_notices,
        "no_registered_charges": no_charges,
    }


def _parse_credit_information(text: str) -> list[str]:
    bullets = []
    lowered = text.lower()
    signals = parse_business_bureau_signals(text)

    score = signals.get("credit_score")
    score_range = signals.get("credit_score_range")
    if score_range:
        bullets.append(f"Credit score: {score_range}")
    elif score is not None:
        bullets.append(f"Credit score: {score}")
    elif signals.get("credit_score_suppressed"):
        bullets.append("Credit score: Suppressed / unavailable")

    credit_limit = signals.get("credit_limit")
    if credit_limit is not None:
        bullets.append(f"Credit limit: {POUND}{credit_limit:,}")

    money_pattern = rf"({re.escape(POUND)}\s*\d[\d,]*|GBP\s*\d[\d,]*)"
    mrc_current = _re_first(rf"\bmax\.?\s*recommended\s*credit\b[\s\S]{{0,100}}?{money_pattern}", text)
    match_from = re.search(
        rf"\bfrom\s+([A-Za-z]{{3,9}}\s+\d{{4}})\s*\(\s*{money_pattern}\s*\)",
        text,
        re.IGNORECASE,
    )
    from_date = match_from.group(1).strip() if match_from else None
    from_amt = match_from.group(2).strip() if match_from else None

    if mrc_current:
        if from_date and from_amt:
            bullets.append(f"Max. recommended credit: {_money_clean(mrc_current)} (from {from_date} - {_money_clean(from_amt)})")
        else:
            bullets.append(f"Max. recommended credit: {_money_clean(mrc_current)}")
    elif from_date and from_amt:
        bullets.append(f"Max. recommended credit: {_money_clean(from_amt)} (from {from_date})")

    enquiries_3m = signals.get("enquiries_3m")
    if enquiries_3m:
        bullets.append(f"Searches: {enquiries_3m} enquiries in last 3 months")

    negative_impact = signals.get("negative_impact_count")
    if "needs attention" in lowered:
        if negative_impact:
            bullets.append(f"Credit risk factors: Needs attention (Negative impact: {negative_impact})")
        else:
            bullets.append("Credit risk factors: Needs attention")

    return bullets


def _parse_payment_performance(text: str) -> list[str]:
    bullets = []
    lowered = text.lower()

    if "unable to get information" in lowered and "payment average" in lowered:
        return ["Payment average: Unavailable"]

    if "payment average" in lowered:
        avg = _re_first(r"\bpayment average\b\s*[\r\n ]+([A-Za-z][A-Za-z\s]+?)(?:\n|$)", text)
        if avg:
            bullets.append(f"Payment average: {avg.strip()}")

    days_late = _re_first(r"\b(\d+)\s+days\s+late\b", text)
    if days_late:
        bullets.append(f"Average days late: {days_late} days late")

    industry = _re_first(r"\bindustry average\b[\s\S]{0,60}\b(\d+)\s+days\s+late\b", text)
    if industry:
        bullets.append(f"Industry average: {industry} days late")

    if "significantly worsening payment pattern" in lowered:
        bullets.append("Pattern: Significantly worsening payment pattern")
    elif "worsening payment pattern" in lowered:
        bullets.append("Pattern: Worsening payment pattern")

    return bullets


def _parse_legal_notices(text: str) -> list[str]:
    bullets = []
    lowered = text.lower()
    legal_text = _section_text(text, ("legal notices",), ("public record", "charges", "credit information", "payment performance")) or text
    legal_lowered = legal_text.lower()

    ccj_yes = bool(re.search(r"county\s+court\s+judg(e)?ment\s+registered", legal_lowered))
    if ccj_yes:
        bullets.append("County Court Judgement registered: Yes")
    elif re.search(r"\bno\s+(?:county\s+court\s+judg(e)?ment|ccj)", legal_lowered):
        bullets.append("County Court Judgement registered: No")

    if "no legal notices registered" in lowered:
        bullets.append("Legal notices registered: No")
        return bullets

    reg_source = _re_first(r"\bregistered\b\s*\n\s*([A-Z][A-Z\s]+)\n", legal_text, flags=0)
    if reg_source:
        bullets.append(f"Registered: {reg_source.strip()}")

    ref = _re_first(r"\bregistered\b[\s\S]{0,200}\b([A-Z0-9]{6,})\b", legal_text)
    if ref and re.match(r"^[A-Z0-9]{6,}$", ref):
        bullets.append(f"Ref: {ref}")

    if ccj_yes:
        date = _re_first(r"\b(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})\b", legal_text)
        if date:
            bullets.append(f"Date: {date}")

    money_values = re.findall(rf"{re.escape(POUND)}\s*\d[\d,]*|GBP\s*\d[\d,]*", legal_text) if ccj_yes else []
    if money_values:
        max_value = max(money_values, key=lambda value: int(re.sub(r"[^\d]", "", value) or "0"))
        bullets.append(f"Value: {_money_clean(max_value)}")

    return bullets


def _parse_public_record(text: str) -> list[str]:
    bullets = []
    if not text:
        return bullets

    full = norm_pdf_text(text)
    header_lines = [line.strip() for line in full.splitlines() if line.strip()]

    company_name = None
    ignored = {"capital report", "credit information", "payment performance", "legal notices", "public record", "charges"}
    for index, line in enumerate(header_lines[:25]):
        if line.lower() == "capital report" and index + 1 < len(header_lines):
            candidate = header_lines[index + 1].strip()
            if candidate and candidate.lower() not in ignored:
                company_name = candidate
                break
    if not company_name:
        for line in header_lines[:25]:
            if line.isupper() and len(line) >= 5 and line.lower() not in ignored:
                company_name = line
                break

    start = None
    for index, line in enumerate(header_lines):
        if line.strip().lower() == "public record":
            start = index
            break

    block = []
    if start is not None:
        end = len(header_lines)
        for index in range(start + 1, len(header_lines)):
            if header_lines[index].strip().lower() in ("charges", "credit information", "payment performance", "legal notices"):
                end = index
                break
        block = header_lines[start + 1:end]

    combined_company_row = None
    for index, line in enumerate(block):
        match = re.match(
            r"^(.+?)\s+(\d{6,10})\s+([A-Za-z]+)\s+(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})$",
            line,
        )
        if match:
            name, number, status, inc_date = match.groups()
            if index + 1 < len(block) and re.match(r"^[A-Z]{2,5}$", block[index + 1]):
                name = f"{name} {block[index + 1]}"
            combined_company_row = {
                "company_name": name.strip(),
                "company_number": number,
                "company_status": status,
                "incorporation_date": inc_date,
            }
            break

    labels = {
        "company number": "Company number",
        "company status": "Company Status",
        "incorporation date": "Incorporation date",
        "sic": "SIC",
    }
    label_set = set(labels.keys())

    def is_label_line(value: str) -> bool:
        lowered = value.strip().lower()
        return any(label in lowered for label in label_set)

    def next_value_after(label: str) -> str | None:
        for index, line in enumerate(block):
            if label.lower() in line.lower():
                tail = re.sub(rf"(?i).*{re.escape(label)}\s*[:\-]?\s*", "", line).strip()
                if tail and not is_label_line(tail):
                    return tail
                for next_index in range(index + 1, min(index + 6, len(block))):
                    candidate = block[next_index].strip()
                    if candidate and not is_label_line(candidate):
                        return candidate
        return None

    company_number = combined_company_row.get("company_number") if combined_company_row else None
    candidate_number = next_value_after("Company number")
    if candidate_number:
        match = re.search(r"\b(\d{6,10})\b", candidate_number)
        if match:
            company_number = match.group(1)
    if not company_number and block:
        match = re.search(r"\b(\d{6,10})\b", "\n".join(block))
        if match:
            company_number = match.group(1)

    company_status = combined_company_row.get("company_status") if combined_company_row else None
    company_status = company_status or next_value_after("Company Status") or next_value_after("Company status")
    if company_status and is_label_line(company_status):
        company_status = None

    incorporation_date = combined_company_row.get("incorporation_date") if combined_company_row else None
    candidate_inc = next_value_after("Incorporation date")
    if not incorporation_date and candidate_inc:
        match = re.search(r"\b(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})\b", candidate_inc)
        if match:
            incorporation_date = match.group(1)
    if not incorporation_date and block:
        match = re.search(r"\b(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})\b", "\n".join(block))
        if match:
            incorporation_date = match.group(1)

    sic = None
    candidate_sic = next_value_after("SIC")
    if candidate_sic:
        match = re.search(r"\b(\d{4,6}\s*-\s*.+)$", candidate_sic)
        if match:
            sic = match.group(1).strip()
    if not sic and block:
        match = re.search(r"\b(\d{4,6}\s*-\s*[A-Za-z].+)", "\n".join(block))
        if match:
            sic = match.group(1).strip()

    if combined_company_row and combined_company_row.get("company_name"):
        company_name = combined_company_row["company_name"]

    if company_name:
        bullets.append(f"Company name: {company_name}")
    if company_number:
        bullets.append(f"Company number: {company_number}")
    if company_status:
        bullets.append(f"Company Status: {company_status.strip()}")
    if incorporation_date:
        bullets.append(f"Incorporation date: {incorporation_date}")
    if sic:
        bullets.append(f"SIC: {sic}")

    return bullets


def _parse_charges(text: str) -> list[str]:
    lowered = text.lower()
    if "no mortgages or charges" in lowered:
        return ["No mortgages or charges"]
    if "charges" in lowered:
        return ["Charges: present (details in report)"]
    return []


def build_report_information(full_text: str) -> dict[str, list[str]]:
    text = norm_pdf_text(full_text)
    return {
        "Credit information": _parse_credit_information(text),
        "Payment performance": _parse_payment_performance(text),
        "Legal notices": _parse_legal_notices(text),
        "Public Record": _parse_public_record(text),
        "Charges": _parse_charges(text),
    }


def bureau_band_from_pdf_text(pdf_text: str) -> tuple[str | None, list[str]]:
    text = norm_pdf_text(pdf_text or "")
    if len(text.strip()) < MIN_EXTRACTED_TEXT_CHARS:
        return None, ["No usable business bureau text extracted"]

    lowered = text.lower()
    reasons: list[str] = []
    signals = parse_business_bureau_signals(text)

    if "maximum risk" in lowered:
        return "D (Very High Risk)", ["Report states: Maximum risk"]

    if signals.get("credit_score_suppressed"):
        reasons.append("Business bureau risk score is suppressed")

    credit_limit = signals.get("credit_limit")
    max_credit = signals.get("max_recommended_credit")
    if credit_limit == 0 and max_credit == 0:
        reasons.append(f"Credit limit and max recommended credit are {POUND}0")

    neg_count = int(signals.get("negative_impact_count") or 0)
    if neg_count:
        reasons.append(f"Negative impact factors: {neg_count}")

    enquiries_3m = signals.get("enquiries_3m")
    if isinstance(enquiries_3m, int) and enquiries_3m >= 3:
        reasons.append(f"Recent company searches: {enquiries_3m} enquiries in 3 months")

    if signals.get("no_ccj_recorded"):
        reasons.append("No CCJs recorded")

    if signals.get("credit_score_suppressed") and credit_limit == 0 and max_credit == 0 and neg_count >= 4:
        return "C (High Risk)", reasons
    if signals.get("credit_score_suppressed") or (credit_limit == 0 and max_credit == 0) or neg_count >= 3:
        return "B (Moderate Risk)", reasons

    grade = None
    match_grade = re.search(r"\bcredit\s+score\b[\s\S]{0,120}\b([a-f])\b", text, re.IGNORECASE)
    if match_grade:
        grade = match_grade.group(1).upper()
        reasons.append(f"Detected credit grade: {grade}")

    if grade in ("F", "E"):
        return "D (Very High Risk)", reasons
    if grade == "D":
        return "C (High Risk)", reasons
    if grade == "C":
        return "B (Moderate Risk)", reasons
    if grade in ("A", "B"):
        return "A (Low Risk)", reasons

    score_high = None
    match_range = re.search(r"\bcredit\s+score\b[\s:]*([0-9]{1,3})\s*[-\u2013\u2014]\s*([0-9]{1,3})\b", text, re.IGNORECASE)
    if match_range:
        score_low = int(match_range.group(1))
        score_high = int(match_range.group(2))
        reasons.append(f"Detected credit score range: {score_low}-{score_high}")

    if score_high is not None:
        if score_high >= 12:
            return "D (Very High Risk)", reasons
        if score_high >= 9:
            return "C (High Risk)", reasons
        if score_high >= 6:
            return "B (Moderate Risk)", reasons
        return "A (Low Risk)", reasons

    adverse_rules = [
        (r"\binsolvenc(y|ies)\b|\bliquidat(e|ion)\b|\badministration\b|\bwinding\s+up\b", "Insolvency / liquidation indicator"),
        (r"\bdefault(s)?\b", "Defaults mentioned"),
        (r"\barrear(s)?\b|\blate\s+payment(s)?\b|\bdelinquen(t|cy)\b", "Arrears / late payments mentioned"),
        (r"\bcollections?\b|\bdebt\s+collection\b", "Collections mentioned"),
        (r"\bdissolved\b|\bstrike\s+off\b", "Dissolved/strike-off mentioned"),
        (r"\bneeds\s+attention\b", "Needs attention flagged"),
    ]
    adverse_hits = 0
    for pattern, label in adverse_rules:
        if re.search(pattern, lowered, re.IGNORECASE):
            adverse_hits += 1
            reasons.append(label)

    if adverse_hits >= 3:
        return "D (Very High Risk)", reasons
    if adverse_hits == 2:
        return "C (High Risk)", reasons
    if adverse_hits == 1:
        return "B (Moderate Risk)", reasons

    reasons.append("No specific indicators extracted")
    return None, reasons


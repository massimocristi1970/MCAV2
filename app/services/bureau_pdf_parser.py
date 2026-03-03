import re
import pdfplumber
import re
from typing import Optional

def derive_band_from_text(text: str) -> Optional[str]:
    """
    Try to derive a bureau 'band' from extracted PDF text.
    Returns a short label like 'A', 'B', 'C', 'D', 'E' or None if not found.
    """
    if not text:
        return None

    t = " ".join(str(text).split())  # normalise whitespace

    # Common patterns like: "Band A", "BAND: B", "Risk Band - C"
    m = re.search(r"\bband\s*[:\-]?\s*([A-E])\b", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Sometimes appears as "Score band A" or "Rating band B"
    m = re.search(r"\b(score|rating)\s*band\s*[:\-]?\s*([A-E])\b", t, flags=re.IGNORECASE)
    if m:
        return m.group(2).upper()

    return None

def parse_bureau_pdf(file_bytes):

    ccj_flag = False
    ccj_count = 0
    bureau_band = "Unknown"

    with pdfplumber.open(file_bytes) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""

    text_lower = text.lower()

    # Explicit positive CCJ detection
    if "county court judgment" in text_lower or "ccj recorded" in text_lower:
        if "no county court judgement" not in text_lower:
            ccj_flag = True

            # Optional count extraction
            matches = re.findall(r"ccj.*?(\d+)", text_lower)
            if matches:
                ccj_count = int(matches[0])


    # Bureau band logic (based on existing derived scoring)
    bureau_band = derive_band_from_text(text)

    return {
        "ccj_flag": ccj_flag,
        "ccj_count": ccj_count,
        "bureau_band": bureau_band
    }
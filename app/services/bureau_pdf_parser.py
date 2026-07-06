from __future__ import annotations

from io import BytesIO
from typing import Any

from app.services.business_bureau_pdf import parse_business_bureau_pdf


def parse_bureau_pdf(file_bytes: bytes | BytesIO) -> dict[str, Any]:
    """Compatibility wrapper around the consolidated business bureau parser."""
    raw = file_bytes.getvalue() if hasattr(file_bytes, "getvalue") else file_bytes
    result = parse_business_bureau_pdf(raw)
    return {
        "ccj_flag": result.business_ccj,
        "ccj_count": 1 if result.business_ccj else 0,
        "bureau_band": result.bureau_band,
        "parse_status": result.parse_status,
        "backend": result.backend,
        "error": result.error,
        "signals": result.signals,
    }
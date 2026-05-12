"""Card terminal statement ingestion and reconciliation services."""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from types import SimpleNamespace
from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd
import pdfplumber
from .payment_provider_registry import detect_providers_in_text, provider_catalog
from .provider_parser_profiles import get_provider_profile


@dataclass
class ParseResult:
    """Normalized parse result for one uploaded terminal statement."""

    filename: str
    provider: str
    parser: str
    merchant_id: Optional[str]
    statement_start: Optional[date]
    statement_end: Optional[date]
    currency: str
    gross_card_sales: float
    refunds_amount: float
    chargebacks_amount: float
    fees_total: float
    transaction_count: int
    confidence: float
    warnings: List[str]
    raw_summary: Dict[str, Any]
    extraction_diagnostics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "provider": self.provider,
            "parser": self.parser,
            "merchant_id": self.merchant_id,
            "statement_start": self.statement_start,
            "statement_end": self.statement_end,
            "currency": self.currency,
            "gross_card_sales": self.gross_card_sales,
            "refunds_amount": self.refunds_amount,
            "chargebacks_amount": self.chargebacks_amount,
            "fees_total": self.fees_total,
            "transaction_count": self.transaction_count,
            "confidence": self.confidence,
            "warnings": self.warnings,
            "raw_summary": self.raw_summary,
            "extraction_diagnostics": self.extraction_diagnostics,
        }


class CardTerminalIngestionService:
    """Ingest card-terminal statements across providers and formats."""

    _money_re = re.compile(r"£?\s*([0-9][0-9,]*\.?[0-9]{0,2})")
    _period_re = re.compile(r"Period:\s*([0-9]{2}/[0-9]{2}/[0-9]{4})\s*-\s*([0-9]{2}/[0-9]{2}/[0-9]{4})", re.IGNORECASE)
    _merchant_re = re.compile(r"Merchant ID\s*:?\s*([0-9]{8,20})", re.IGNORECASE)
    _currency_re = re.compile(r"Currency:\s*([A-Z]{3})", re.IGNORECASE)
    _clover_transactions_summary_re = re.compile(
        r"Your Card Processing Statement.*?Description\s+Amount.*?Transactions\s+([0-9][0-9,]*\.?[0-9]{0,2})",
        re.IGNORECASE | re.DOTALL,
    )
    _clover_total_row_re = re.compile(
        r"Transactions\s+Currency:\s*[A-Z]{3}.*?Card type.*?Total\s+([0-9]{1,7})\s+([0-9][0-9,]*\.?[0-9]{0,2})",
        re.IGNORECASE | re.DOTALL,
    )

    def parse_uploaded_files(self, uploaded_files: List[Any]) -> Dict[str, Any]:
        """Parse multiple uploaded files and return normalized results (includes ZIP expansion)."""
        results: List[ParseResult] = []
        errors: List[Dict[str, str]] = []

        for file in uploaded_files or []:
            filename = getattr(file, "name", "unknown_file")
            try:
                ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
                raw = file.getvalue() if hasattr(file, "getvalue") else file.read()
                if ext == "zip":
                    inner_results, inner_errors = self._parse_zip_archive(filename, raw)
                    results.extend(inner_results)
                    errors.extend(inner_errors)
                else:
                    results.append(self._parse_single(file))
            except Exception as exc:  # pragma: no cover - safety path
                errors.append({"filename": filename, "error": str(exc)})

        df = pd.DataFrame([r.to_dict() for r in results]) if results else pd.DataFrame()
        return {
            "records": results,
            "dataframe": df,
            "errors": errors,
            "parsed_count": len(results),
            "error_count": len(errors),
        }

    def summarize_by_month(self, parsed_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate normalized terminal data by statement month."""
        if parsed_df is None or parsed_df.empty:
            return pd.DataFrame()

        work = parsed_df.copy()
        work["statement_end"] = pd.to_datetime(work["statement_end"], errors="coerce")
        work = work.dropna(subset=["statement_end"])
        if work.empty:
            return pd.DataFrame()

        work["year_month"] = work["statement_end"].dt.to_period("M").astype(str)
        out = (
            work.groupby("year_month", as_index=False)
            .agg(
                gross_card_sales=("gross_card_sales", "sum"),
                refunds_amount=("refunds_amount", "sum"),
                chargebacks_amount=("chargebacks_amount", "sum"),
                fees_total=("fees_total", "sum"),
                transaction_count=("transaction_count", "sum"),
                statements=("filename", "count"),
            )
            .sort_values("year_month")
        )
        return out

    def compare_with_banking_data(self, bank_df: pd.DataFrame, terminal_summary_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compare monthly card sales from terminal statements against bank inflows.
        Uses revenue-like credits as proxy for settled card inflows.
        """
        if bank_df is None or bank_df.empty or terminal_summary_df is None or terminal_summary_df.empty:
            return {"comparison": pd.DataFrame(), "summary": {}}

        b = bank_df.copy()
        b["date"] = pd.to_datetime(b["date"], errors="coerce")
        b = b.dropna(subset=["date"])
        if b.empty:
            return {"comparison": pd.DataFrame(), "summary": {}}

        # Ensure categorization flags exist
        if "is_revenue" not in b.columns:
            b["is_revenue"] = b["amount"] < 0

        # Credits are negative in this project convention; use absolute values
        revenue_credits = b[b["is_revenue"] & (b["amount"] < 0)].copy()
        revenue_credits["year_month"] = revenue_credits["date"].dt.to_period("M").astype(str)
        bank_monthly = (
            revenue_credits.groupby("year_month", as_index=False)
            .agg(bank_revenue_inflows=("amount", lambda s: abs(float(s.sum()))))
            .sort_values("year_month")
        )

        # Provider-aware bank inflow extraction using narration aliases from registry
        narration_cols = [c for c in ["name", "transaction_name", "merchant_name", "name_y"] if c in revenue_credits.columns]
        if narration_cols:
            revenue_credits["narration"] = revenue_credits[narration_cols].fillna("").astype(str).agg(" ".join, axis=1)
        else:
            revenue_credits["narration"] = ""
        revenue_credits["detected_providers"] = revenue_credits["narration"].apply(detect_providers_in_text)
        provider_tx = revenue_credits[revenue_credits["detected_providers"].map(len) > 0].copy()
        if not provider_tx.empty:
            provider_tx = provider_tx.explode("detected_providers")
            provider_bank_monthly = (
                provider_tx.groupby(["year_month", "detected_providers"], as_index=False)
                .agg(provider_bank_inflows=("amount", lambda s: abs(float(s.sum()))))
                .rename(columns={"detected_providers": "provider"})
                .sort_values(["year_month", "provider"])
            )
            provider_detected_inflows_total = float(provider_tx["amount"].abs().sum())
        else:
            provider_bank_monthly = pd.DataFrame(columns=["year_month", "provider", "provider_bank_inflows"])
            provider_detected_inflows_total = 0.0

        terminal_monthly = terminal_summary_df[["year_month", "gross_card_sales", "fees_total", "transaction_count"]].copy()
        merged = pd.merge(terminal_monthly, bank_monthly, on="year_month", how="outer").fillna(0.0)
        merged["difference_amount"] = merged["bank_revenue_inflows"] - merged["gross_card_sales"]
        merged["difference_pct_vs_terminal"] = merged.apply(
            lambda r: (r["difference_amount"] / r["gross_card_sales"] * 100.0) if r["gross_card_sales"] > 0 else 0.0,
            axis=1,
        )
        merged["abs_difference_pct"] = merged["difference_pct_vs_terminal"].abs()

        if len(merged) > 0:
            avg_abs_var = float(merged["abs_difference_pct"].mean())
            coverage = float((merged["gross_card_sales"] > 0).sum() / len(merged) * 100.0)
        else:
            avg_abs_var = 0.0
            coverage = 0.0

        if avg_abs_var <= 15:
            quality = "Good"
        elif avg_abs_var <= 30:
            quality = "Moderate"
        else:
            quality = "Poor"

        return {
            "comparison": merged.sort_values("year_month"),
            "provider_bank_monthly": provider_bank_monthly,
            "summary": {
                "months_compared": int(len(merged)),
                "average_abs_variance_pct": round(avg_abs_var, 2),
                "coverage_pct": round(coverage, 1),
                "reconciliation_quality": quality,
                "providers_detected_in_bank_narration": sorted(provider_tx["detected_providers"].unique().tolist()) if not provider_tx.empty else [],
                "provider_detected_inflows_total": round(provider_detected_inflows_total, 2),
                "provider_detection_coverage_pct": round(
                    (provider_detected_inflows_total / float(revenue_credits["amount"].abs().sum()) * 100.0)
                    if not revenue_credits.empty and float(revenue_credits["amount"].abs().sum()) > 0
                    else 0.0,
                    2,
                ),
                "provider_catalog": provider_catalog(),
            },
        }

    def _parse_zip_archive(self, zip_filename: str, raw: bytes) -> tuple[List[ParseResult], List[Dict[str, str]]]:
        """Expand a ZIP and parse each supported file inside."""
        out: List[ParseResult] = []
        errs: List[Dict[str, str]] = []
        allowed = {"pdf", "csv", "xls", "xlsx"}
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    inner = info.filename.replace("\\", "/")
                    if inner.endswith("/") or inner.startswith("__MACOSX/"):
                        continue
                    inner_base = inner.rsplit("/", 1)[-1]
                    if inner_base.startswith("."):
                        continue
                    ext = inner_base.lower().rsplit(".", 1)[-1] if "." in inner_base else ""
                    if ext not in allowed:
                        continue
                    try:
                        member_bytes = zf.read(info)
                        label = f"{zip_filename}/{inner}"
                        pseudo = SimpleNamespace(name=label, getvalue=lambda b=member_bytes: b)
                        out.append(self._parse_single(pseudo))
                    except Exception as exc:
                        errs.append({"filename": f"{zip_filename}/{inner}", "error": str(exc)})
        except zipfile.BadZipFile as exc:
            errs.append({"filename": zip_filename, "error": f"Invalid ZIP: {exc}"})
        return out, errs

    def _parse_single(self, file: Any) -> ParseResult:
        filename = getattr(file, "name", "unknown_file")
        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
        raw = file.getvalue() if hasattr(file, "getvalue") else file.read()

        if ext == "pdf":
            text = self._extract_pdf_text(raw)
            if "clover" in text.lower() or "merchantportal.app" in text.lower():
                return self._parse_clover_pdf(filename, text)
            if self._is_stripe_balance_report_pdf(text):
                return self._parse_stripe_balance_report_pdf(filename, text)
            if self._is_paypal_merchant_statement_pdf(text):
                return self._parse_paypal_merchant_statement_pdf(filename, text)
            provider_hint = self._detect_provider_hint(f"{filename} {text[:8000]}")
            if provider_hint:
                prof = get_provider_profile(provider_hint)
                if prof:
                    return self._parse_profile_pdf(filename, text, prof)
            return self._parse_generic_pdf(filename, text)

        if ext in {"csv"}:
            df = pd.read_csv(io.BytesIO(raw))
            provider_hint = self._detect_provider_hint(f"{filename} {' '.join([str(c) for c in df.columns[:30]])}")
            prof = get_provider_profile(provider_hint) if provider_hint else None
            return self._parse_tabular_statement(filename, df, provider_hint="csv", profile=prof)

        if ext in {"xls", "xlsx"}:
            df = pd.read_excel(io.BytesIO(raw))
            provider_hint = self._detect_provider_hint(f"{filename} {' '.join([str(c) for c in df.columns[:30]])}")
            prof = get_provider_profile(provider_hint) if provider_hint else None
            return self._parse_tabular_statement(filename, df, provider_hint="excel", profile=prof)

        raise ValueError(f"Unsupported file format: {ext or 'unknown'}")

    def _extract_pdf_text(self, raw: bytes) -> str:
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            return "\n".join((page.extract_text() or "") for page in pdf.pages)

    _stripe_balance_report_markers = re.compile(
        r"balance\s+summary|balance\s+report|about\s+this\s+report",
        re.IGNORECASE,
    )

    def _is_stripe_balance_report_pdf(self, text: str) -> bool:
        t = (text or "").lower()
        if "stripe" not in t:
            return False
        return bool(self._stripe_balance_report_markers.search(text or ""))

    def _parse_stripe_balance_report_pdf(self, filename: str, text: str) -> ParseResult:
        """Stripe Balance / balance-summary PDF (UK-style date range + Charges block)."""
        warnings: List[str] = []
        period_start: Optional[date] = None
        period_end: Optional[date] = None

        # Date range 1 Apr 2026 → 30 Apr 2026 (arrow may be Unicode)
        dr = re.search(
            r"Date\s+range\s+(\d{1,2}\s+\w+\s+\d{4})\s*(?:→|->|—|--|\u2192)\s*(\d{1,2}\s+\w+\s+\d{4})",
            text,
            re.IGNORECASE,
        )
        if dr:
            s = pd.to_datetime(dr.group(1), dayfirst=True, errors="coerce")
            e = pd.to_datetime(dr.group(2), dayfirst=True, errors="coerce")
            period_start = s.date() if pd.notna(s) else None
            period_end = e.date() if pd.notna(e) else None
        if period_end is None:
            warnings.append("Could not parse Stripe statement date range.")

        acct = None
        m_acct = re.search(r"\(acct_([A-Za-z0-9]+)\)", text)
        if m_acct:
            acct = f"acct_{m_acct.group(1)}"

        gross = 0.0
        txn_count = 0
        # Allow a non-digit glyph (e.g. £ garbled in extraction) between ) and the amount.
        m_ch = re.search(r"Charges\s*\((\d+)\)\s*[^\d-]*([\d,]+\.?\d*)", text, re.IGNORECASE)
        if m_ch:
            txn_count = int(m_ch.group(1))
            gross = self._to_float(m_ch.group(2))

        refunds = 0.0
        m_rf = re.search(r"Refunds\s*\((\d+)\)\s*[^\d-]*([\d,]+\.?\d*)", text, re.IGNORECASE)
        if m_rf:
            refunds = abs(self._to_float(m_rf.group(2)))

        fees = 0.0
        # Prefer the summary "Activity" block (fees on balance activity), not payout-side lines like
        # "Payout fees" / "Fees £12.06" under the Payouts table.
        act_block = re.search(r"(?ms)^Activity\s*\n^(.*?)^\s*Payouts\s*$", text)
        if act_block:
            ab = act_block.group(1)
            m_fee = re.search(r"(?mi)^Fees\s+-\s*(?:£|\u00a3)?\s*([\d,]+\.\d{2})", ab)
            if not m_fee:
                m_fee = re.search(r"(?mi)^Fees\s+-\D*?([\d,]+\.\d{2})", ab)
            if m_fee:
                fees = abs(self._to_float(m_fee.group(1)))
        if fees == 0.0:
            # Detailed breakdown: primary "Fees" plus "Additional Stripe fees" (same period).
            det = re.search(
                r"(?ms)^Balance\s+change\s+from\s+activity\s*$(.*?)(?=^About\s+this\s+report\s*$)",
                text,
                re.IGNORECASE,
            )
            if det:
                sub = det.group(1)
                fee_parts: List[float] = []
                for m in re.finditer(
                    r"(?mi)^Fees\s+-\s*(?:£|\u00a3)?\s*([\d,]+\.\d{2})",
                    sub,
                ):
                    fee_parts.append(abs(self._to_float(m.group(1))))
                for m in re.finditer(
                    r"(?mi)^Additional\s+Stripe\s+fees\s*\(\d+\)\s+-\s*(?:£|\u00a3)?\s*([\d,]+\.\d{2})",
                    sub,
                ):
                    fee_parts.append(abs(self._to_float(m.group(1))))
                if fee_parts:
                    fees = float(sum(fee_parts))

        chargebacks = 0.0
        currency = "GBP" if "£" in text else (self._extract_first(self._currency_re, text) or "GBP")

        has_charges_line = m_ch is not None
        confidence = (
            0.92
            if period_end is not None and (gross > 0 or fees > 0 or (has_charges_line and txn_count == 0))
            else 0.65
        )
        if not has_charges_line:
            warnings.append("Stripe balance report: could not find Charges(N) line.")
        warnings.append("Parsed as Stripe Balance Report PDF (native).")

        return ParseResult(
            filename=filename,
            provider="Stripe",
            parser="stripe_balance_report_pdf_v1",
            merchant_id=acct,
            statement_start=period_start,
            statement_end=period_end,
            currency=currency,
            gross_card_sales=float(gross),
            refunds_amount=float(refunds),
            chargebacks_amount=float(chargebacks),
            fees_total=float(fees),
            transaction_count=int(txn_count),
            confidence=confidence,
            warnings=warnings,
            raw_summary={"text_length": len(text)},
            extraction_diagnostics={
                "profile": "Stripe",
                "fields": {
                    "gross_card_sales": has_charges_line,
                    "fees_total": fees > 0,
                    "period_end": period_end is not None,
                    "merchant_id": acct is not None,
                },
                "fallback_used": not has_charges_line and gross <= 0 and fees <= 0,
            },
        )

    def _is_paypal_merchant_statement_pdf(self, text: str) -> bool:
        t = text or ""
        if "paypal" not in t.lower() and "pay pal" not in t.lower():
            return False
        return bool(re.search(r"Merchant\s+Account\s+ID", t, re.IGNORECASE)) and bool(
            re.search(r"Activity\s+Summary", t, re.IGNORECASE)
        )

    def _parse_paypal_merchant_statement_pdf(self, filename: str, text: str) -> ParseResult:
        """PayPal merchant monthly statement PDF (Activity Summary + Payments received)."""
        warnings: List[str] = []
        period_start: Optional[date] = None
        period_end: Optional[date] = None

        m_per = re.search(
            r"Activity\s+Summary\s*\((\d{2}/\d{2}/\d{4})\s*-\s*(\d{2}/\d{2}/\d{4})\)",
            text,
            re.IGNORECASE,
        )
        if m_per:
            s = pd.to_datetime(m_per.group(1), dayfirst=True, errors="coerce")
            e = pd.to_datetime(m_per.group(2), dayfirst=True, errors="coerce")
            period_start = s.date() if pd.notna(s) else None
            period_end = e.date() if pd.notna(e) else None
        if period_end is None:
            m2 = re.search(r"(\d{2}/\d{2}/\d{4})\s*-\s*(\d{2}/\d{2}/\d{4})", text)
            if m2:
                s = pd.to_datetime(m2.group(1), dayfirst=True, errors="coerce")
                e = pd.to_datetime(m2.group(2), dayfirst=True, errors="coerce")
                period_start = s.date() if pd.notna(s) else period_start
                period_end = e.date() if pd.notna(e) else period_end

        merchant_id = None
        m_mid = re.search(r"Merchant\s+Account\s+ID:\s*([A-Z0-9]+)", text, re.IGNORECASE)
        if m_mid:
            merchant_id = m_mid.group(1).strip()

        gross = 0.0
        m_pay = re.search(r"Payments\s+received\s+([\d,]+\.?\d*)", text, re.IGNORECASE)
        if m_pay:
            gross = abs(self._to_float(m_pay.group(1)))

        fees = 0.0
        m_fee = re.search(r"^Fees\s+-?([\d,]+\.?\d*)", text, re.IGNORECASE | re.MULTILINE)
        if m_fee:
            fees = abs(self._to_float(m_fee.group(1)))

        refunds = 0.0
        m_ref = re.search(r"Payments\s+sent\s+-?([\d,]+\.?\d*)", text, re.IGNORECASE)
        if m_ref:
            refunds = abs(self._to_float(m_ref.group(1)))

        chargebacks = 0.0
        txn_count = len(re.findall(r"\bID:\s+[A-Z0-9]{10,20}\b", text))

        currency = "GBP"
        confidence = 0.9 if gross > 0 and period_end is not None else 0.65
        if gross <= 0:
            warnings.append("PayPal statement: could not find Payments received amount.")
        warnings.append("Parsed as PayPal Merchant Statement PDF (native).")

        return ParseResult(
            filename=filename,
            provider="PayPal",
            parser="paypal_merchant_statement_pdf_v1",
            merchant_id=merchant_id,
            statement_start=period_start,
            statement_end=period_end,
            currency=currency,
            gross_card_sales=float(gross),
            refunds_amount=float(refunds),
            chargebacks_amount=float(chargebacks),
            fees_total=float(fees),
            transaction_count=int(txn_count),
            confidence=confidence,
            warnings=warnings,
            raw_summary={"text_length": len(text)},
            extraction_diagnostics={
                "profile": "PayPal",
                "fields": {
                    "gross_card_sales": gross > 0,
                    "fees_total": fees > 0,
                    "period_end": period_end is not None,
                    "merchant_id": merchant_id is not None,
                },
                "fallback_used": gross <= 0,
            },
        )

    def _parse_clover_pdf(self, filename: str, text: str) -> ParseResult:
        merchant_id = self._extract_first(self._merchant_re, text)
        period_start, period_end = self._extract_period(text)
        currency = self._extract_first(self._currency_re, text) or "GBP"

        gross_card_sales = self._extract_clover_gross_sales(text)
        fees_total = self._extract_amount_after_label(text, "Total Fees (excluding VAT)")
        if fees_total == 0.0:
            fees_total = self._extract_amount_after_label(text, "Transaction Fees")
        refunds_amount = self._extract_amount_after_label(text, "Refunds")
        chargebacks_amount = self._extract_amount_after_label(text, "Chargebacks / Reversals")
        transaction_count = self._extract_clover_total_txn_count(text)

        warnings: List[str] = []
        if gross_card_sales <= 0:
            warnings.append("Could not confidently extract gross card sales from statement.")
        if period_end is None:
            warnings.append("Could not extract statement period.")

        confidence = 0.95 if gross_card_sales > 0 and period_end is not None else 0.7
        return ParseResult(
            filename=filename,
            provider="Clover",
            parser="clover_pdf_v1",
            merchant_id=merchant_id,
            statement_start=period_start,
            statement_end=period_end,
            currency=currency,
            gross_card_sales=float(gross_card_sales),
            refunds_amount=float(refunds_amount),
            chargebacks_amount=float(chargebacks_amount),
            fees_total=float(fees_total),
            transaction_count=int(transaction_count),
            confidence=confidence,
            warnings=warnings,
            raw_summary={"text_length": len(text)},
            extraction_diagnostics={
                "profile": "Clover",
                "fields": {
                    "gross_card_sales": gross_card_sales > 0,
                    "fees_total": fees_total > 0,
                    "period_end": period_end is not None,
                    "merchant_id": merchant_id is not None,
                },
                "fallback_used": gross_card_sales <= 0,
            },
        )

    def _parse_generic_pdf(self, filename: str, text: str) -> ParseResult:
        merchant_id = self._extract_first(self._merchant_re, text)
        period_start, period_end = self._extract_period(text)
        currency = self._extract_first(self._currency_re, text) or "GBP"

        gross_card_sales = self._extract_amount_after_label(text, "Transactions")
        fees_total = self._extract_amount_after_label(text, "Transaction Fees")
        refunds_amount = self._extract_amount_after_label(text, "Refunds")
        chargebacks_amount = self._extract_amount_after_label(text, "Chargeback")

        warnings = [
            "Parsed with generic PDF parser; validate totals before relying operationally."
        ]
        confidence = 0.6 if gross_card_sales > 0 else 0.45
        provider = self._detect_provider_hint(f"{filename} {text[:2000]}")
        return ParseResult(
            filename=filename,
            provider=provider or "Unknown",
            parser="generic_pdf_v1",
            merchant_id=merchant_id,
            statement_start=period_start,
            statement_end=period_end,
            currency=currency,
            gross_card_sales=float(gross_card_sales),
            refunds_amount=float(refunds_amount),
            chargebacks_amount=float(chargebacks_amount),
            fees_total=float(fees_total),
            transaction_count=0,
            confidence=confidence,
            warnings=warnings,
            raw_summary={"text_length": len(text)},
            extraction_diagnostics={
                "profile": provider or "Unknown",
                "fields": {
                    "gross_card_sales": gross_card_sales > 0,
                    "fees_total": fees_total > 0,
                    "period_end": period_end is not None,
                    "merchant_id": merchant_id is not None,
                },
                "fallback_used": True,
            },
        )

    def _parse_profile_pdf(self, filename: str, text: str, profile) -> ParseResult:
        merchant_id = self._extract_first(self._merchant_re, text)
        period_start, period_end = self._extract_period(text)
        currency = self._extract_first(self._currency_re, text) or "GBP"

        gross_card_sales = self._extract_from_labels(text, profile.amount_labels_pdf)
        fees_total = self._extract_from_labels(text, profile.fee_labels_pdf)
        refunds_amount = self._extract_from_labels(text, profile.refund_labels_pdf)
        chargebacks_amount = self._extract_from_labels(text, profile.chargeback_labels_pdf)
        transaction_count = self._extract_total_txn_count(text)

        confidence = 0.8 if gross_card_sales > 0 else 0.55
        warnings: List[str] = []
        if gross_card_sales <= 0:
            warnings.append("Native profile used, but gross card sales could not be confidently extracted.")
        warnings.append("Native provider profile parser used (layout may vary by statement template/version).")

        return ParseResult(
            filename=filename,
            provider=profile.provider,
            parser=f"{profile.provider.lower().replace(' ', '_')}_profile_pdf_v1",
            merchant_id=merchant_id,
            statement_start=period_start,
            statement_end=period_end,
            currency=currency,
            gross_card_sales=float(gross_card_sales),
            refunds_amount=float(refunds_amount),
            chargebacks_amount=float(chargebacks_amount),
            fees_total=float(fees_total),
            transaction_count=int(transaction_count),
            confidence=confidence,
            warnings=warnings,
            raw_summary={"text_length": len(text)},
            extraction_diagnostics={
                "profile": profile.provider,
                "fields": {
                    "gross_card_sales": gross_card_sales > 0,
                    "fees_total": fees_total > 0,
                    "period_end": period_end is not None,
                    "merchant_id": merchant_id is not None,
                },
                "fallback_used": gross_card_sales <= 0,
            },
        )

    def _parse_tabular_statement(self, filename: str, df: pd.DataFrame, provider_hint: str, profile=None) -> ParseResult:
        normalized_cols = {str(c).strip().lower(): c for c in df.columns}
        warnings: List[str] = []

        amount_candidates = profile.amount_columns_tabular if profile else ["amount", "gross", "sales", "sale amount", "transaction amount"]
        date_candidates = profile.date_columns_tabular if profile else ["date", "transaction date", "settlement date"]
        fee_candidates = profile.fee_columns_tabular if profile else ["fee", "fees", "charge", "commission"]
        refund_candidates = profile.refund_columns_tabular if profile else ["refund", "refund amount"]
        merchant_candidates = profile.merchant_columns_tabular if profile else ["merchant id", "merchant", "mid"]
        chargeback_candidates = profile.chargeback_columns_tabular if profile else ["chargeback", "reversal"]

        amount_col = self._pick_col(normalized_cols, amount_candidates)
        date_col = self._pick_col(normalized_cols, date_candidates)
        fee_col = self._pick_col(normalized_cols, fee_candidates)
        refund_col = self._pick_col(normalized_cols, refund_candidates)
        merchant_col = self._pick_col(normalized_cols, merchant_candidates)
        chargeback_col = self._pick_col(normalized_cols, chargeback_candidates)

        if amount_col is None:
            raise ValueError("Unable to identify sales amount column in tabular statement.")

        work = df.copy()
        work[amount_col] = pd.to_numeric(work[amount_col], errors="coerce").fillna(0.0)
        gross = float(work[amount_col].clip(lower=0).sum())
        tx_count = int((work[amount_col] > 0).sum())

        fees_total = float(pd.to_numeric(work[fee_col], errors="coerce").fillna(0.0).abs().sum()) if fee_col else 0.0
        refunds = float(pd.to_numeric(work[refund_col], errors="coerce").fillna(0.0).abs().sum()) if refund_col else 0.0
        chargebacks = float(pd.to_numeric(work[chargeback_col], errors="coerce").fillna(0.0).abs().sum()) if chargeback_col else 0.0
        merchant_id = str(work[merchant_col].dropna().iloc[0]) if merchant_col and not work[merchant_col].dropna().empty else None

        start_dt: Optional[date] = None
        end_dt: Optional[date] = None
        if date_col:
            work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
            valid_dates = work[date_col].dropna()
            if not valid_dates.empty:
                start_dt = valid_dates.min().date()
                end_dt = valid_dates.max().date()
        else:
            warnings.append("No date column detected; monthly alignment may be less reliable.")

        provider_probe_text = f"{filename} {' '.join([str(c) for c in df.columns[:20]])}"
        provider = self._detect_provider_hint(provider_probe_text)
        parser_name = f"generic_{provider_hint}_v1"
        conf = 0.75 if end_dt else 0.6
        if profile:
            parser_name = f"{profile.provider.lower().replace(' ', '_')}_profile_{provider_hint}_v1"
            conf = 0.85 if end_dt else 0.7
            warnings.append("Native provider profile parser used for tabular statement mapping.")
        return ParseResult(
            filename=filename,
            provider=provider or "Unknown",
            parser=parser_name,
            merchant_id=merchant_id,
            statement_start=start_dt,
            statement_end=end_dt,
            currency="GBP",
            gross_card_sales=gross,
            refunds_amount=refunds,
            chargebacks_amount=chargebacks,
            fees_total=fees_total,
            transaction_count=tx_count,
            confidence=conf,
            warnings=warnings,
            raw_summary={"columns": list(df.columns)},
            extraction_diagnostics={
                "profile": (profile.provider if profile else "Generic"),
                "fields": {
                    "gross_card_sales": gross > 0,
                    "fees_total": fees_total > 0 if fee_col else False,
                    "period_end": end_dt is not None,
                    "merchant_id": merchant_id is not None,
                    "amount_col_detected": amount_col is not None,
                    "date_col_detected": date_col is not None,
                },
                "fallback_used": profile is None,
            },
        )

    @staticmethod
    def _extract_first(pattern: re.Pattern, text: str) -> Optional[str]:
        match = pattern.search(text or "")
        if not match:
            return None
        return match.group(1).strip()

    def _extract_period(self, text: str) -> tuple[Optional[date], Optional[date]]:
        match = self._period_re.search(text or "")
        if not match:
            return None, None
        start = pd.to_datetime(match.group(1), dayfirst=True, errors="coerce")
        end = pd.to_datetime(match.group(2), dayfirst=True, errors="coerce")
        return (
            start.date() if pd.notna(start) else None,
            end.date() if pd.notna(end) else None,
        )

    def _extract_amount_after_label(self, text: str, label: str) -> float:
        if not text:
            return 0.0
        pattern = re.compile(re.escape(label) + r"[^\n\r£0-9]*([£]?\s*[0-9][0-9,]*\.?[0-9]{0,2})", re.IGNORECASE)
        match = pattern.search(text)
        if match:
            return self._to_float(match.group(1))

        # fallback: line-based relaxed matching
        for line in text.splitlines():
            if label.lower() in line.lower():
                all_matches = self._money_re.findall(line)
                if all_matches:
                    return self._to_float(all_matches[-1])
        return 0.0

    def _extract_total_txn_count(self, text: str) -> int:
        # Example line: "Total 1005 10,615.99 0 0.00"
        for line in text.splitlines():
            if line.strip().lower().startswith("total "):
                nums = re.findall(r"\b[0-9]{1,6}\b", line)
                if nums:
                    try:
                        return int(nums[0])
                    except ValueError:
                        return 0
        return 0

    def _extract_clover_gross_sales(self, text: str) -> float:
        """
        Clover-specific gross sales extraction.
        Prioritizes the statement summary/table sections and avoids matching
        unrelated values like "last 12 months".
        """
        if not text:
            return 0.0

        m = self._clover_transactions_summary_re.search(text)
        if m:
            return self._to_float(m.group(1))

        m = self._clover_total_row_re.search(text)
        if m:
            return self._to_float(m.group(2))

        # Fallback to generic extraction as a last resort
        return self._extract_amount_after_label(text, "Transactions")

    def _extract_clover_total_txn_count(self, text: str) -> int:
        """Extract Clover transaction count from the card-type total row."""
        if not text:
            return 0

        m = self._clover_total_row_re.search(text)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return 0

        # Fallback to generic extraction
        return self._extract_total_txn_count(text)

    def _extract_from_labels(self, text: str, labels: List[str]) -> float:
        for label in labels or []:
            value = self._extract_amount_after_label(text, label)
            if value > 0:
                return value
        return 0.0

    @staticmethod
    def _detect_provider_hint(text: str) -> Optional[str]:
        hits = detect_providers_in_text(text or "")
        return hits[0] if hits else None

    @staticmethod
    def _to_float(value: Any) -> float:
        if value is None:
            return 0.0
        s = str(value).replace("£", "").replace(",", "").strip()
        try:
            return float(s)
        except ValueError:
            return 0.0

    @staticmethod
    def _pick_col(normalized_cols: Dict[str, Any], candidates: List[str]) -> Optional[Any]:
        for c in candidates:
            for normalized, original in normalized_cols.items():
                if c == normalized or c in normalized:
                    return original
        return None

"""Card terminal statement ingestion and reconciliation services."""

from __future__ import annotations

import io
import re
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from calendar import monthrange
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pdfplumber

try:
    from PyPDF2 import PdfReader as _PyPdfReader
except ImportError:  # pragma: no cover
    _PyPdfReader = None

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
                    results.extend(self._parse_file_results(file))
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

    def derive_card_processing_insights(
        self,
        parsed_df: pd.DataFrame,
        monthly_terminal_df: pd.DataFrame,
        comparison_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Derive review/export signals from card processor statements."""
        if parsed_df is None or parsed_df.empty:
            return {
                "Card Processing Insight Layer": "Not available",
                "Card Processing Insights Used In Score": "No - analysis/export only",
            }

        monthly = monthly_terminal_df.copy() if monthly_terminal_df is not None else pd.DataFrame()
        comp = comparison_payload.get("comparison", pd.DataFrame()) if comparison_payload else pd.DataFrame()
        summary = comparison_payload.get("summary", {}) if comparison_payload else {}

        gross_total = float(pd.to_numeric(parsed_df.get("gross_card_sales", 0), errors="coerce").fillna(0).sum())
        refunds_total = float(pd.to_numeric(parsed_df.get("refunds_amount", 0), errors="coerce").fillna(0).sum())
        chargebacks_total = float(pd.to_numeric(parsed_df.get("chargebacks_amount", 0), errors="coerce").fillna(0).sum())
        fees_total = float(pd.to_numeric(parsed_df.get("fees_total", 0), errors="coerce").fillna(0).sum())
        txn_count = int(pd.to_numeric(parsed_df.get("transaction_count", 0), errors="coerce").fillna(0).sum())

        if monthly is not None and not monthly.empty:
            sales = pd.to_numeric(monthly.get("gross_card_sales", pd.Series(dtype="float64")), errors="coerce").fillna(0)
            active_sales = sales[sales > 0]
            monthly_avg = float(active_sales.mean()) if not active_sales.empty else 0.0
            weakest_month = float(active_sales.min()) if not active_sales.empty else 0.0
            strongest_month = float(active_sales.max()) if not active_sales.empty else 0.0
            volatility = float(active_sales.std(ddof=0) / monthly_avg) if len(active_sales) > 1 and monthly_avg else 0.0
            latest_sales = float(sales.iloc[-1]) if len(sales) else 0.0
            previous_avg = float(sales.iloc[:-1].mean()) if len(sales) > 1 else 0.0
            latest_drop_pct = float((previous_avg - latest_sales) / previous_avg) if previous_avg and latest_sales < previous_avg else 0.0
            months_present = int((sales > 0).sum())
        else:
            monthly_avg = weakest_month = strongest_month = volatility = latest_sales = latest_drop_pct = 0.0
            months_present = 0

        refund_ratio = refunds_total / gross_total if gross_total else 0.0
        chargeback_ratio = chargebacks_total / gross_total if gross_total else 0.0
        fee_ratio = fees_total / gross_total if gross_total else 0.0
        avg_transaction_value = gross_total / txn_count if txn_count else 0.0

        if comp is not None and not comp.empty:
            terminal_total = float(pd.to_numeric(comp.get("gross_card_sales", 0), errors="coerce").fillna(0).sum())
            bank_revenue_total = float(pd.to_numeric(comp.get("bank_revenue_inflows", 0), errors="coerce").fillna(0).sum())
            unmatched_shortfall = float(
                comp.apply(
                    lambda r: max(float(r.get("gross_card_sales", 0) or 0) - float(r.get("bank_revenue_inflows", 0) or 0), 0.0),
                    axis=1,
                ).sum()
            )
            card_vs_ob_revenue_ratio = terminal_total / bank_revenue_total if bank_revenue_total else 0.0
            unmatched_pct = unmatched_shortfall / terminal_total if terminal_total else 0.0
        else:
            bank_revenue_total = 0.0
            card_vs_ob_revenue_ratio = 0.0
            unmatched_shortfall = 0.0
            unmatched_pct = 0.0

        concerns: List[str] = []
        positives: List[str] = []

        reconciliation_quality = summary.get("reconciliation_quality", "N/A")
        if reconciliation_quality == "Good":
            positives.append("Card sales reconcile well to bank revenue")
        elif reconciliation_quality == "Poor":
            concerns.append("Poor reconciliation between card statements and bank revenue")

        if refund_ratio >= 0.10:
            concerns.append("High refund ratio")
        elif gross_total and refund_ratio <= 0.05:
            positives.append("Low refund ratio")

        if chargeback_ratio >= 0.01:
            concerns.append("Elevated chargeback ratio")
        elif gross_total and chargeback_ratio == 0:
            positives.append("No chargebacks detected in uploaded statements")

        if volatility >= 0.45:
            concerns.append("Volatile card sales")
        elif months_present >= 3 and volatility <= 0.25:
            positives.append("Stable card sales")

        if latest_drop_pct >= 0.30:
            concerns.append("Latest card sales month materially below prior average")

        if unmatched_pct >= 0.25:
            concerns.append("Material card sales shortfall versus bank revenue evidence")

        if chargeback_ratio >= 0.02 or unmatched_pct >= 0.40 or latest_drop_pct >= 0.50:
            suitability = "Weak"
        elif concerns:
            suitability = "Review"
        elif positives:
            suitability = "Strong"
        else:
            suitability = "Acceptable"

        return {
            "Card Processing Insight Layer": "Available",
            "Card Processing Insights Used In Score": "No - analysis/export only",
            "Card Processor Statements Parsed": int(len(parsed_df)),
            "Card Processor Months Present": months_present,
            "Card Sales Total": round(gross_total, 2),
            "Card Sales Monthly Average": round(monthly_avg, 2),
            "Card Weakest Month Sales": round(weakest_month, 2),
            "Card Strongest Month Sales": round(strongest_month, 2),
            "Card Latest Month Sales": round(latest_sales, 2),
            "Card Sales Volatility": round(volatility, 3),
            "Card Latest Month Drop Pct": round(latest_drop_pct, 3),
            "Card Refund Ratio": round(refund_ratio, 3),
            "Card Chargeback Ratio": round(chargeback_ratio, 3),
            "Card Fee Ratio": round(fee_ratio, 3),
            "Card Average Transaction Value": round(avg_transaction_value, 2),
            "Card Transaction Count": txn_count,
            "Card vs OB Revenue Ratio": round(card_vs_ob_revenue_ratio, 3),
            "Card Unmatched Sales Shortfall": round(unmatched_shortfall, 2),
            "Card Unmatched Sales Shortfall Pct": round(unmatched_pct, 3),
            "Card Reconciliation Quality": reconciliation_quality,
            "Card MCA Suitability": suitability,
            "Card Processing Positive Signals": positives,
            "Card Processing Concerns": concerns,
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
                        out.extend(self._parse_file_results(pseudo))
                    except Exception as exc:
                        errs.append({"filename": f"{zip_filename}/{inner}", "error": str(exc)})
        except zipfile.BadZipFile as exc:
            errs.append({"filename": zip_filename, "error": f"Invalid ZIP: {exc}"})
        return out, errs

    def _parse_single(self, file: Any) -> ParseResult:
        """Parse one upload; when a file expands to many months, returns the first record."""
        results = self._parse_file_results(file)
        if not results:
            raise ValueError("No parse results produced for file.")
        return results[0]

    def _parse_file_results(self, file: Any) -> List[ParseResult]:
        filename = getattr(file, "name", "unknown_file")
        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
        raw = file.getvalue() if hasattr(file, "getvalue") else file.read()

        if ext == "pdf":
            sample = self._extract_pdf_text_pages(raw, max_pages=2)
            if self._is_shopify_sales_transactions_pdf(sample, filename):
                return self._parse_shopify_sales_transactions_from_pdf(filename, raw)
            text = self._extract_pdf_text(raw)
            if "clover" in text.lower() or "merchantportal.app" in text.lower():
                return [self._parse_clover_pdf(filename, text)]
            if self._is_stripe_balance_report_pdf(text):
                return [self._parse_stripe_balance_report_pdf(filename, text)]
            if self._is_paypal_merchant_statement_pdf(text):
                return [self._parse_paypal_merchant_statement_pdf(filename, text)]
            if self._is_paypal_transaction_history_pdf(text):
                return [self._parse_paypal_transaction_history_pdf(filename, text)]
            if self._is_zempler_business_statement_pdf(text, filename):
                return [self._parse_zempler_business_statement_pdf(filename, text)]
            if self._is_dojo_invoice_pdf(text, filename):
                return [self._parse_dojo_invoice_pdf(filename, text)]
            if self._is_dna_payments_statement_pdf(text, filename):
                return [self._parse_dna_payments_statement_pdf(filename, text)]
            if self._is_mypos_monthly_statement_pdf(text, filename):
                return [self._parse_mypos_monthly_statement_pdf(filename, text)]
            provider_hint = self._detect_provider_hint(f"{filename} {text[:8000]}")
            if provider_hint:
                prof = get_provider_profile(provider_hint)
                if prof:
                    return [self._parse_profile_pdf(filename, text, prof)]
            generic = self._parse_generic_pdf(filename, text)
            if generic.gross_card_sales <= 0 and self._is_dna_payments_statement_pdf(text, filename):
                return [self._parse_dna_payments_statement_pdf(filename, text)]
            if generic.gross_card_sales <= 0 and self._is_mypos_monthly_statement_pdf(text, filename):
                return [self._parse_mypos_monthly_statement_pdf(filename, text)]
            return [generic]

        if ext in {"csv"}:
            df = pd.read_csv(io.BytesIO(raw))
            if self._is_shopify_sales_transactions_frame(df, filename):
                return self._parse_shopify_sales_transactions_frame(filename, df)
            provider_hint = self._detect_provider_hint(f"{filename} {' '.join([str(c) for c in df.columns[:30]])}")
            prof = get_provider_profile(provider_hint) if provider_hint else None
            return [self._parse_tabular_statement(filename, df, provider_hint="csv", profile=prof)]

        if ext in {"xls", "xlsx"}:
            df = pd.read_excel(io.BytesIO(raw))
            if self._is_shopify_sales_transactions_frame(df, filename):
                return self._parse_shopify_sales_transactions_frame(filename, df)
            provider_hint = self._detect_provider_hint(f"{filename} {' '.join([str(c) for c in df.columns[:30]])}")
            prof = get_provider_profile(provider_hint) if provider_hint else None
            return [self._parse_tabular_statement(filename, df, provider_hint="excel", profile=prof)]

        raise ValueError(f"Unsupported file format: {ext or 'unknown'}")

    @staticmethod
    def _normalize_pdf_text(text: str) -> str:
        """Clean PDF extraction artifacts and join common label/value line breaks."""
        t = (text or "").replace("\x00", " ")
        joins = (
            (r"(Processed\s+volume,\s*GBP)\s*\n\s*", r"\1 "),
            (r"(NET\s+Processed\s+volume,\s*GBP)\s*\n\s*", r"\1 "),
            (r"(Refunds\s+and\s+chargebacks\s+volume,\s*GBP)\s*\n\s*", r"\1 "),
            (r"(A\s+Transactional\s+fees)\s*\n\s*-?\s*", r"\1 -"),
            (r"(Billing\s+period)\s*\n\s*", r"\1 "),
            (r"(Statement\s+date)\s*\n\s*", r"\1 "),
        )
        for pattern, repl in joins:
            t = re.sub(pattern, repl, t, flags=re.IGNORECASE)
        return t

    def _extract_pdf_text_pypdf2(self, raw: bytes) -> str:
        if _PyPdfReader is None:
            return ""
        try:
            reader = _PyPdfReader(io.BytesIO(raw))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception:
            return ""

    def _extract_pdf_text_pages(self, raw: bytes, max_pages: Optional[int] = None) -> str:
        parts: List[str] = []
        try:
            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                pages = pdf.pages if max_pages is None else pdf.pages[:max_pages]
                for page in pages:
                    parts.append(page.extract_text() or "")
        except Exception:
            return ""
        return self._normalize_pdf_text("\n".join(parts))

    def _extract_pdf_text(self, raw: bytes) -> str:
        text = self._extract_pdf_text_pages(raw)
        if len(text.strip()) < 200:
            alt = self._normalize_pdf_text(self._extract_pdf_text_pypdf2(raw))
            if len(alt.strip()) > len(text.strip()):
                text = alt
        return text

    _shopify_tx_line_re = re.compile(
        r"^(\d{2})/(\d{2})/(\d{4})\s+(\d+)\s+(\d+)(.+)$",
    )

    def _is_shopify_sales_transactions_pdf(self, text: str, filename: str = "") -> bool:
        fn = (filename or "").lower()
        if "shopify" in fn and ("sales" in fn or "transaction" in fn):
            return True
        t = text or ""
        return bool(
            re.search(r"Month\s+Sale\s+ID\s+Order\s+number", t, re.IGNORECASE)
            and re.search(r"Gross\s+sales", t, re.IGNORECASE)
            and re.search(r"Net\s+sales", t, re.IGNORECASE)
        )

    @staticmethod
    def _normalize_shopify_columns(columns: List[Any]) -> List[str]:
        return [re.sub(r"\s+", " ", str(c or "").strip().lower()) for c in columns]

    def _is_shopify_sales_transactions_frame(self, df: pd.DataFrame, filename: str = "") -> bool:
        if df is None or df.empty:
            return False
        fn = (filename or "").lower()
        cols = self._normalize_shopify_columns(list(df.columns))
        colset = set(cols)
        has_cols = {"gross sales", "net sales"}.issubset(colset) or (
            "gross sales" in " ".join(cols) and "net sales" in " ".join(cols)
        )
        if "shopify" in fn and has_cols:
            return True
        return bool(
            has_cols
            and any("sale id" in c for c in cols)
            and any(c in ("month", "day", "sale date") or "date" in c for c in cols)
        )

    def _shopify_accumulate_line(self, line: str, by_month: Dict[str, Dict[str, Any]]) -> None:
        line = (line or "").strip()
        if not line or re.search(r"^Month\s+Sale\s+ID", line, re.IGNORECASE):
            return
        m = self._shopify_tx_line_re.match(line)
        if not m:
            return
        _dd, mm, yyyy = m.group(1), m.group(2), m.group(3)
        rest = m.group(6)
        nums = re.findall(r"-?[\d,]+\.?\d*", rest)
        if len(nums) < 8:
            return
        gross, _disc, returns, net, _ship, _retfee, _tax, _total = (
            self._to_float(x) for x in nums[-8:]
        )
        ym = f"{yyyy}-{mm}"
        bucket = by_month.setdefault(
            ym,
            {
                "gross": 0.0,
                "net": 0.0,
                "refunds": 0.0,
                "count": 0,
                "year": int(yyyy),
                "month": int(mm),
            },
        )
        bucket["gross"] += gross
        bucket["net"] += net
        bucket["refunds"] += abs(returns)
        bucket["count"] += 1

    def _shopify_month_period(self, year: int, month: int) -> Tuple[date, date]:
        last_day = monthrange(year, month)[1]
        return date(year, month, 1), date(year, month, last_day)

    def _shopify_results_from_monthly(
        self, filename: str, by_month: Dict[str, Dict[str, Any]]
    ) -> List[ParseResult]:
        if not by_month:
            return [
                ParseResult(
                    filename=filename,
                    provider="Shopify Payments",
                    parser="shopify_sales_transactions_pdf_v1",
                    merchant_id=None,
                    statement_start=None,
                    statement_end=None,
                    currency="GBP",
                    gross_card_sales=0.0,
                    refunds_amount=0.0,
                    chargebacks_amount=0.0,
                    fees_total=0.0,
                    transaction_count=0,
                    confidence=0.45,
                    warnings=[
                        "Shopify Sales-Transactions report: no transaction rows found.",
                    ],
                    raw_summary={"months": 0},
                    extraction_diagnostics={
                        "profile": "Shopify Payments",
                        "fields": {
                            "gross_card_sales": False,
                            "fees_total": False,
                            "period_end": False,
                            "merchant_id": False,
                        },
                        "fallback_used": True,
                    },
                )
            ]

        base_name = filename.rsplit("/", 1)[-1]
        results: List[ParseResult] = []
        for ym in sorted(by_month.keys()):
            bucket = by_month[ym]
            period_start, period_end = self._shopify_month_period(bucket["year"], bucket["month"])
            gross = float(bucket["gross"])
            txn_count = int(bucket["count"])
            warnings = [
                "Parsed as Shopify Sales-Transactions report (native): one row per calendar month.",
                "Processing/payout fees are not in this export; fees_total is set to 0.",
            ]
            results.append(
                ParseResult(
                    filename=f"{base_name} ({ym})",
                    provider="Shopify Payments",
                    parser="shopify_sales_transactions_pdf_v1",
                    merchant_id=None,
                    statement_start=period_start,
                    statement_end=period_end,
                    currency="GBP",
                    gross_card_sales=gross,
                    refunds_amount=float(bucket["refunds"]),
                    chargebacks_amount=0.0,
                    fees_total=0.0,
                    transaction_count=txn_count,
                    confidence=0.9 if gross > 0 else 0.55,
                    warnings=warnings,
                    raw_summary={
                        "year_month": ym,
                        "net_sales": float(bucket["net"]),
                        "source_file": filename,
                    },
                    extraction_diagnostics={
                        "profile": "Shopify Payments",
                        "fields": {
                            "gross_card_sales": gross > 0,
                            "fees_total": False,
                            "period_end": True,
                            "merchant_id": False,
                        },
                        "fallback_used": False,
                    },
                )
            )
        return results

    def _parse_shopify_sales_transactions_from_pdf(
        self, filename: str, raw: bytes
    ) -> List[ParseResult]:
        by_month: Dict[str, Dict[str, Any]] = {}
        try:
            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    for line in page_text.splitlines():
                        self._shopify_accumulate_line(line, by_month)
        except Exception:
            return self._shopify_results_from_monthly(filename, {})
        return self._shopify_results_from_monthly(filename, by_month)

    def _parse_shopify_sales_transactions_frame(
        self, filename: str, df: pd.DataFrame
    ) -> List[ParseResult]:
        work = df.copy()
        work.columns = self._normalize_shopify_columns(list(work.columns))

        def pick_col(options: List[str]) -> Optional[str]:
            for opt in options:
                if opt in work.columns:
                    return opt
            for col in work.columns:
                if any(opt in col for opt in options):
                    return col
            return None

        month_col = pick_col(["month"])
        gross_col = pick_col(["gross sales"])
        net_col = pick_col(["net sales"])
        returns_col = pick_col(["returns"])
        if not gross_col or not month_col:
            return self._shopify_results_from_monthly(filename, {})

        work["_gross"] = pd.to_numeric(
            work[gross_col].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        ).fillna(0.0)
        work["_net"] = (
            pd.to_numeric(
                work[net_col].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            ).fillna(0.0)
            if net_col
            else work["_gross"]
        )
        work["_returns"] = (
            pd.to_numeric(
                work[returns_col].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            ).fillna(0.0)
            .abs()
            if returns_col
            else 0.0
        )
        work["_month"] = pd.to_datetime(work[month_col], dayfirst=True, errors="coerce")
        work = work.dropna(subset=["_month"])
        if work.empty:
            return self._shopify_results_from_monthly(filename, {})

        by_month: Dict[str, Dict[str, Any]] = {}
        for _, row in work.iterrows():
            ts = row["_month"]
            ym = f"{ts.year:04d}-{ts.month:02d}"
            bucket = by_month.setdefault(
                ym,
                {
                    "gross": 0.0,
                    "net": 0.0,
                    "refunds": 0.0,
                    "count": 0,
                    "year": int(ts.year),
                    "month": int(ts.month),
                },
            )
            bucket["gross"] += float(row["_gross"])
            bucket["net"] += float(row["_net"])
            bucket["refunds"] += float(row["_returns"])
            bucket["count"] += 1

        return self._shopify_results_from_monthly(filename, by_month)

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

    def _is_paypal_transaction_history_pdf(self, text: str) -> bool:
        t = text or ""
        if "paypal" not in t.lower() and "express checkout payment" not in t.lower():
            return False
        return bool(re.search(r"Transaction\s+History", t, re.IGNORECASE)) and bool(
            re.search(r"Date\s+Description\s+Status\s+Currency\s+Gross\s+Fee\s+Net", t, re.IGNORECASE)
        )

    def _parse_paypal_transaction_history_pdf(self, filename: str, text: str) -> ParseResult:
        """PayPal transaction-history PDF export (row-level Gross/Fee/Net table)."""
        warnings: List[str] = []
        period_start: Optional[date] = None
        period_end: Optional[date] = None

        m_per = re.search(
            r"([A-Z][a-z]+\s+\d{1,2},\s+\d{4})\s+through\s+([A-Z][a-z]+\s+\d{1,2},\s+\d{4})",
            text,
            re.IGNORECASE,
        )
        if m_per:
            s = pd.to_datetime(m_per.group(1), dayfirst=False, errors="coerce")
            e = pd.to_datetime(m_per.group(2), dayfirst=False, errors="coerce")
            period_start = s.date() if pd.notna(s) else None
            period_end = e.date() if pd.notna(e) else None

        merchant_id = None
        m_email = re.search(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", text, re.IGNORECASE)
        if m_email:
            merchant_id = m_email.group(0).strip()

        row_re = re.compile(
            r"(?ms)^(\d{2}/\d{2}/\d{4})\s+(.+?)\s+ID:\s+([A-Z0-9]+)\s+"
            r"(Completed|Pending|Refused|Denied|Cancelled|Canceled)\s+([A-Z]{3})\s+"
            r"(-?[\d,]+\.\d{2})\s+(-?[\d,]+\.\d{2})\s+(-?[\d,]+\.\d{2})",
            re.IGNORECASE,
        )
        sale_markers = (
            "express checkout payment",
            "paypal checkout payment",
            "website payment",
            "mobile payment",
            "payment received",
        )
        sale_rows: Dict[str, tuple[float, float]] = {}
        currencies: List[str] = []
        for match in row_re.finditer(text or ""):
            desc = re.sub(r"\s+", " ", match.group(2)).strip().lower()
            txn_id = match.group(3).strip()
            status = match.group(4).strip().lower()
            currency = match.group(5).strip().upper()
            gross = self._to_float(match.group(6))
            fee = self._to_float(match.group(7))

            if status != "completed" or gross <= 0:
                continue
            if not any(marker in desc for marker in sale_markers):
                continue
            if any(skip in desc for skip in ("hold", "withdrawal", "authorisation", "authorization", "cashback")):
                continue
            sale_rows.setdefault(txn_id, (gross, fee))
            currencies.append(currency)

        gross_total = round(sum(gross for gross, _fee in sale_rows.values()), 2)
        fees_total = round(sum(abs(fee) for _gross, fee in sale_rows.values() if fee < 0), 2)
        txn_count = len(sale_rows)
        currency = currencies[0] if currencies else "GBP"

        if gross_total <= 0:
            warnings.append("PayPal transaction history: could not find completed customer payment rows.")
        if period_end is None:
            warnings.append("Could not parse PayPal transaction-history period.")
        warnings.append("Parsed as PayPal Transaction History PDF (native).")

        confidence = 0.9 if gross_total > 0 and period_end is not None else 0.65
        return ParseResult(
            filename=filename,
            provider="PayPal",
            parser="paypal_transaction_history_pdf_v1",
            merchant_id=merchant_id,
            statement_start=period_start,
            statement_end=period_end,
            currency=currency,
            gross_card_sales=float(gross_total),
            refunds_amount=0.0,
            chargebacks_amount=0.0,
            fees_total=float(fees_total),
            transaction_count=int(txn_count),
            confidence=confidence,
            warnings=warnings,
            raw_summary={"text_length": len(text), "sales_transaction_ids": sorted(sale_rows.keys())},
            extraction_diagnostics={
                "profile": "PayPal",
                "fields": {
                    "gross_card_sales": gross_total > 0,
                    "fees_total": fees_total > 0,
                    "period_end": period_end is not None,
                    "merchant_id": merchant_id is not None,
                },
                "fallback_used": gross_total <= 0,
            },
        )

    def _is_dojo_invoice_pdf(self, text: str, filename: str = "") -> bool:
        t = text or ""
        fn = (filename or "").lower()
        if "dojo" in fn and re.search(r"Invoice\s+period", t, re.IGNORECASE):
            return True
        return bool(
            re.search(r"Your\s+Dojo\s+invoice", t, re.IGNORECASE)
            and re.search(r"Card\s+transaction\s+rates", t, re.IGNORECASE)
            and re.search(r"Invoice\s+period", t, re.IGNORECASE)
        )

    def _parse_dojo_invoice_pdf(self, filename: str, text: str) -> ParseResult:
        """Dojo monthly invoice PDF (card transaction rates total = gross card sales)."""
        warnings: List[str] = []
        period_start: Optional[date] = None
        period_end: Optional[date] = None

        m_per = re.search(
            r"Invoice\s+period\s+(\d{1,2}\s+\w+\s+\d{4})\s+to\s+(\d{1,2}\s+\w+\s+\d{4})",
            text,
            re.IGNORECASE,
        )
        if m_per:
            start_ts = pd.to_datetime(m_per.group(1), dayfirst=True, errors="coerce")
            end_ts = pd.to_datetime(m_per.group(2), dayfirst=True, errors="coerce")
            if pd.notna(start_ts):
                period_start = start_ts.date()
            if pd.notna(end_ts):
                period_end = end_ts.date()
        if period_end is None:
            warnings.append("Could not parse Dojo invoice period.")

        merchant_id = None
        m_mid = re.search(r"Merchant\s+ID\s+(\d+)", text, re.IGNORECASE)
        if m_mid:
            merchant_id = m_mid.group(1).strip()

        gross = 0.0
        txn_count = 0
        m_rates = re.search(
            r"Card\s+transaction\s+rates[\s\S]{0,2500}?^Total\s+([\d,]+)\s+£([\d,]+\.?\d*)\s+£([\d,]+\.?\d*)",
            text,
            re.IGNORECASE | re.MULTILINE,
        )
        if m_rates:
            txn_count = int(self._to_float(m_rates.group(1)))
            gross = self._to_float(m_rates.group(2))

        fees = 0.0
        for label in ("Card transaction fees", "Card transaction rates", "Additional rates"):
            m_fee = re.search(rf"{re.escape(label)}\s+£([\d,]+\.?\d*)", text, re.IGNORECASE)
            if m_fee:
                fees += self._to_float(m_fee.group(1))

        refunds = 0.0
        chargebacks = 0.0
        currency = "GBP"

        if gross <= 0:
            warnings.append("Dojo invoice: could not find Card transaction rates total volume.")
        warnings.append(
            "Parsed as Dojo invoice PDF (native): gross is total card transaction value; "
            "fees sum secure transaction, interchange-style rates, and additional rates "
            "(excludes card machine & account services)."
        )

        confidence = 0.92 if gross > 0 and period_end is not None else 0.65

        return ParseResult(
            filename=filename,
            provider="Dojo",
            parser="dojo_invoice_pdf_v1",
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
                "profile": "Dojo",
                "fields": {
                    "gross_card_sales": gross > 0,
                    "fees_total": fees > 0,
                    "period_end": period_end is not None,
                    "merchant_id": merchant_id is not None,
                },
                "fallback_used": gross <= 0,
            },
        )

    def _is_dna_payments_statement_pdf(self, text: str, filename: str = "") -> bool:
        t = text or ""
        fn = (filename or "").lower().replace("\\", "/").rsplit("/", 1)[-1]
        fn_key = fn.replace(" ", "").replace("_", "")
        if "dnapayment" in fn_key:
            return True
        # DNA merchant portal exports often use numeric PDF filenames.
        if re.search(r"^\d{10,}\.pdf$", fn) and re.search(
            r"Merchant\s+Processing|Statement\s+ID\s+\d{10,}", t, re.IGNORECASE
        ):
            return True
        markers = [
            bool(re.search(r"Merchant\s+Processing\s+Statement", t, re.IGNORECASE)),
            bool(
                re.search(
                    r"DNA\s+Payments|dnapayments\.|DNA\s+Payments\s+Limited",
                    t,
                    re.IGNORECASE,
                )
            ),
            bool(re.search(r"Processed\s+volume", t, re.IGNORECASE)),
            bool(re.search(r"Statement\s+ID\s+\d{10,}", t, re.IGNORECASE)),
        ]
        return sum(markers) >= 2

    def _parse_dna_payments_statement_pdf(self, filename: str, text: str) -> ParseResult:
        """DNA Payments merchant processing statement PDF."""
        warnings: List[str] = []
        period_start: Optional[date] = None
        period_end: Optional[date] = None

        m_bill = re.search(r"Billing\s+period\s+(\w+)\s+(\d{4})", text, re.IGNORECASE)
        if m_bill:
            month_ts = pd.to_datetime(f"1 {m_bill.group(1)} {m_bill.group(2)}", errors="coerce")
            if pd.notna(month_ts):
                period_start = month_ts.date()
                period_end = (month_ts + pd.offsets.MonthEnd(0)).date()
        if period_end is None:
            m_stmt = re.search(r"Statement\s+date\s+(\d{1,2}\s+\w+\s+\d{4})", text, re.IGNORECASE)
            if m_stmt:
                stmt_ts = pd.to_datetime(m_stmt.group(1), dayfirst=True, errors="coerce")
                if pd.notna(stmt_ts):
                    period_end = (stmt_ts - pd.Timedelta(days=1)).date()

        merchant_id = None
        m_mid = re.search(r"Merchant\s+ID\s+(\d+)", text, re.IGNORECASE)
        if m_mid:
            merchant_id = m_mid.group(1).strip()

        gross = 0.0
        for vol_pat in (
            r"Processed\s+volume,\s*GBP\s*([\d,]+\.?\d*)",
            r"NET\s+Processed\s+volume,\s*GBP\s*([\d,]+\.?\d*)",
            r"Processed\s+volume,\s*GBP\s*\n\s*([\d,]+\.?\d*)",
        ):
            m_vol = re.search(vol_pat, text, re.IGNORECASE)
            if m_vol:
                gross = self._to_float(m_vol.group(1))
                break

        refunds = 0.0
        m_ref = re.search(
            r"Refunds\s+and\s+chargebacks\s+volume,\s*GBP\s*([\d,]+\.?\d*)",
            text,
            re.IGNORECASE,
        )
        if m_ref:
            refunds = self._to_float(m_ref.group(1))

        fees = 0.0
        for fee_pat in (
            r"A\s+Transactional\s+fees\s+-?\s*([\d,]+\.?\d*)",
            r"Transactional\s+fees\s+-?\s*([\d,]+\.?\d*)",
        ):
            m_fee = re.search(fee_pat, text, re.IGNORECASE)
            if m_fee:
                fees = abs(self._to_float(m_fee.group(1)))
                break

        txn_count = 0
        m_tot = re.search(
            r"^Totals\s+([\d,]+\.?\d*)\s+([\d,]+)\s+-?([\d,]+\.?\d*)",
            text,
            re.IGNORECASE | re.MULTILINE,
        )
        if m_tot:
            txn_count = int(self._to_float(m_tot.group(2)))
            if gross <= 0:
                gross = self._to_float(m_tot.group(1))
            if fees <= 0:
                fees = abs(self._to_float(m_tot.group(3)))

        chargebacks = 0.0
        currency = "GBP"

        if gross <= 0:
            warnings.append("DNA Payments statement: could not find processed volume.")
        warnings.append("Parsed as DNA Payments Merchant Processing Statement PDF (native).")

        confidence = 0.92 if gross > 0 and period_end is not None else 0.65

        return ParseResult(
            filename=filename,
            provider="DNA Payments",
            parser="dna_payments_merchant_statement_pdf_v1",
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
                "profile": "DNA Payments",
                "fields": {
                    "gross_card_sales": gross > 0,
                    "fees_total": fees > 0,
                    "period_end": period_end is not None,
                    "merchant_id": merchant_id is not None,
                },
                "fallback_used": gross <= 0,
            },
        )

    def _is_mypos_monthly_statement_pdf(self, text: str, filename: str = "") -> bool:
        """myPOS monthly payment report PDF (Report summary + Gross payments)."""
        fn = (filename or "").lower().replace("\\", "/")
        if any(k in fn for k in ("payment_report_monthly", "my_pos", "mypos")):
            if re.search(r"Monthly\s+statement\s*-\s*\d{2}\.\d{4}", text or "", re.IGNORECASE):
                return True
        t = text or ""
        return bool(
            re.search(r"Monthly\s+statement\s*-\s*\d{2}\.\d{4}", t, re.IGNORECASE)
            and re.search(r"Gross\s+payments:\s*[\d,]+", t, re.IGNORECASE)
            and re.search(r"myPOS\s+Payments", t, re.IGNORECASE)
        )

    def _parse_mypos_monthly_statement_pdf(self, filename: str, text: str) -> ParseResult:
        """myPOS monthly payment report: summary totals for card + online payments."""
        warnings: List[str] = []
        period_start: Optional[date] = None
        period_end: Optional[date] = None

        m_per = re.search(r"Monthly\s+statement\s*-\s*(\d{2})\.(\d{4})", text, re.IGNORECASE)
        if m_per:
            month = int(m_per.group(1))
            year = int(m_per.group(2))
            if 1 <= month <= 12:
                period_start = date(year, month, 1)
                period_end = date(year, month, monthrange(year, month)[1])
        if period_end is None:
            warnings.append("Could not parse myPOS statement period (Monthly statement - MM.YYYY).")

        merchant_id = None
        m_acct = re.search(r"Account\s+number:\s*(\d+)", text, re.IGNORECASE)
        if m_acct:
            merchant_id = m_acct.group(1).strip()

        company_name = None
        m_name = re.search(r"Name\s+(.+)", text)
        if m_name:
            company_name = m_name.group(1).strip()

        gross = 0.0
        m_gross = re.search(
            r"Gross\s+payments:\s*([\d,]+\.?\d*)\s*(?:GBP)?",
            text,
            re.IGNORECASE,
        )
        if m_gross:
            gross = self._to_float(m_gross.group(1))

        fees = 0.0
        m_fee = re.search(
            r"Total\s+fees:\s*([\d,]+\.?\d*)\s*(?:GBP)?",
            text,
            re.IGNORECASE,
        )
        if m_fee:
            fees = self._to_float(m_fee.group(1))

        net = 0.0
        m_net = re.search(
            r"Net\s+payments:\s*([\d,]+\.?\d*)\s*(?:GBP)?",
            text,
            re.IGNORECASE,
        )
        if m_net:
            net = self._to_float(m_net.group(1))

        refunds = 0.0
        summary_block = text.split("Transactions", 1)[0] if "Transactions" in text else text[:4000]
        for m_ref in re.finditer(r"^Refunds\s+([\d,]+\.?\d*)", summary_block, re.IGNORECASE | re.MULTILINE):
            refunds += self._to_float(m_ref.group(1))

        card_machine = 0.0
        m_cm = re.search(r"Card\s+machine\s+payments\s+([\d,]+\.?\d*)", text, re.IGNORECASE)
        if m_cm:
            card_machine = self._to_float(m_cm.group(1))

        online = 0.0
        m_on = re.search(r"Online\s+payments\s+([\d,]+\.?\d*)", text, re.IGNORECASE)
        if m_on:
            online = self._to_float(m_on.group(1))

        txn_count = len(re.findall(r"myPOS\s+Payment", text, re.IGNORECASE))
        chargebacks = 0.0
        currency = "GBP"

        if gross <= 0:
            warnings.append("myPOS report: could not find Gross payments total.")
        warnings.append(
            "Parsed as myPOS monthly payment report PDF (native): gross is total card + online payments."
        )

        confidence = 0.92 if gross > 0 and period_end is not None else 0.65

        return ParseResult(
            filename=filename,
            provider="myPOS",
            parser="mypos_monthly_payment_report_pdf_v1",
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
            raw_summary={
                "text_length": len(text),
                "company_name": company_name,
                "net_payments": net,
                "card_machine_payments": card_machine,
                "online_payments": online,
            },
            extraction_diagnostics={
                "profile": "myPOS",
                "fields": {
                    "gross_card_sales": gross > 0,
                    "fees_total": fees > 0,
                    "period_end": period_end is not None,
                    "merchant_id": merchant_id is not None,
                },
                "fallback_used": gross <= 0,
            },
        )

    _zempler_statement_row_re = re.compile(
        r"^(\d{2}/\d{2}/\d{4})\s+(\d{4})\s+(.+?)\s+(-?£[\d,]+\.?\d*)\s+(-?£[\d,]+\.?\d*)\s*$",
        re.MULTILINE,
    )

    def _is_zempler_business_statement_pdf(self, text: str, filename: str = "") -> bool:
        """Zempler (formerly Cashplus) business account PDF with card settlement lines."""
        t = text or ""
        fn = (filename or "").lower()
        if "zempler" in fn or "zemplar" in fn:
            if re.search(r"From\s+\d{2}/\d{2}/\d{4}\s+to\s+\d{2}/\d{2}/\d{4}", t, re.IGNORECASE):
                return True
        has_header = bool(
            re.search(r"Date\s+Card\s+ending\s+in\s+Description\s+Amount\s+Balance", t, re.IGNORECASE)
        )
        has_period = bool(re.search(r"From\s+\d{2}/\d{2}/\d{4}\s+to\s+\d{2}/\d{2}/\d{4}", t, re.IGNORECASE))
        has_account = bool(re.search(r"Business\s+Account", t, re.IGNORECASE))
        has_sort = "08-71-99" in t or "087199" in re.sub(r"\s+", "", t)
        return has_header and has_period and has_account and has_sort

    @staticmethod
    def _is_zempler_card_settlement_credit(description: str, amount: float) -> bool:
        """Positive credits that represent card/acquirer remittances on a Zempler business account."""
        if amount <= 0:
            return False
        d = description or ""
        if re.search(r"YL\s*III\s+Limited.*FND", d, re.IGNORECASE):
            return False
        if re.search(r"US\s+Bank\s+Europe", d, re.IGNORECASE):
            return True
        if re.search(r"YouLend\s+Limited.*\bEMS\d", d, re.IGNORECASE):
            return True
        if re.search(r"YouLend\s+Limited\s+YL\d+OUT", d, re.IGNORECASE):
            return True
        if re.search(r"YouLend\s+Limited.*SumUp\s+payment", d, re.IGNORECASE):
            return True
        if re.search(r"Sent\s+from\s+SumUp", d, re.IGNORECASE):
            return True
        return False

    @staticmethod
    def _is_zempler_card_fee(description: str, amount: float) -> bool:
        if amount >= 0:
            return False
        if "Electronic Payment Fee" in (description or ""):
            return True
        return (description or "").strip() == "Annual Fee"

    def _parse_zempler_transaction_rows(self, text: str) -> List[tuple[str, float]]:
        rows: List[tuple[str, float]] = []
        for m in self._zempler_statement_row_re.finditer(text or ""):
            rows.append((m.group(3).strip(), self._to_float(m.group(4))))
        return rows

    def _parse_zempler_business_statement_pdf(self, filename: str, text: str) -> ParseResult:
        """Zempler business account PDF: aggregate card settlement/remittance credits and payment fees."""
        warnings: List[str] = []
        period_start: Optional[date] = None
        period_end: Optional[date] = None

        m_per = re.search(
            r"From\s+(\d{2}/\d{2}/\d{4})\s+to\s+(\d{2}/\d{2}/\d{4})",
            text,
            re.IGNORECASE,
        )
        if m_per:
            s = pd.to_datetime(m_per.group(1), dayfirst=True, errors="coerce")
            e = pd.to_datetime(m_per.group(2), dayfirst=True, errors="coerce")
            period_start = s.date() if pd.notna(s) else None
            period_end = e.date() if pd.notna(e) else None
        if period_end is None:
            warnings.append("Could not parse Zempler statement period (From … to …).")

        merchant_id = None
        m_acct = re.search(r"Account\s+number:\s*(\d+)", text, re.IGNORECASE)
        if m_acct:
            merchant_id = m_acct.group(1).strip()
        m_co = re.search(r"Account\s+held\s+under\s+company\s+name:\s*(.+)", text, re.IGNORECASE)
        company_name = m_co.group(1).strip() if m_co else None

        rows = self._parse_zempler_transaction_rows(text)
        gross = 0.0
        fees = 0.0
        txn_count = 0
        for desc, amt in rows:
            if self._is_zempler_card_settlement_credit(desc, amt):
                gross += amt
                txn_count += 1
            elif self._is_zempler_card_fee(desc, amt):
                fees += abs(amt)

        currency = "GBP"
        if gross <= 0 and txn_count == 0:
            warnings.append("No card settlement credits found (EMS / YouLend OUT / SumUp remittance lines).")
        warnings.append(
            "Parsed as Zempler business account PDF (native): gross is the sum of card "
            "settlement/remittance credits; Kal Pay and internal transfers are excluded."
        )

        confidence = 0.9 if gross > 0 and period_end is not None else 0.65

        return ParseResult(
            filename=filename,
            provider="Zempler",
            parser="zempler_business_account_pdf_v1",
            merchant_id=merchant_id,
            statement_start=period_start,
            statement_end=period_end,
            currency=currency,
            gross_card_sales=float(gross),
            refunds_amount=0.0,
            chargebacks_amount=0.0,
            fees_total=float(fees),
            transaction_count=int(txn_count),
            confidence=confidence,
            warnings=warnings,
            raw_summary={
                "text_length": len(text),
                "company_name": company_name,
                "transaction_rows_parsed": len(rows),
            },
            extraction_diagnostics={
                "profile": "Zempler",
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

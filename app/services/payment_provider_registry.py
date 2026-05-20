"""Registry of payment providers and settlement narration aliases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import re
from .provider_parser_profiles import providers_with_native_profiles


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    aliases: List[str]
    docs_url: Optional[str] = None
    notes: Optional[str] = None


# Built from existing categorization engine terms + UK market coverage.
PROVIDER_SPECS: List[ProviderSpec] = [
    ProviderSpec("Clover", ["clover", "fiserv", "first data"], "https://www.clover.com/"),
    ProviderSpec("Stripe", ["stripe"], "https://stripe.com/docs/reports/api"),
    ProviderSpec("SumUp", ["sumup", "sum up"], "https://developer.sumup.com/api"),
    ProviderSpec(
        "Zempler",
        ["zempler", "zemplar", "cashplus", "087199", "08-71-99"],
        "https://zemplerbank.com/",
        notes="Business account PDF; card settlements appear as EMS / YouLend / SumUp credits.",
    ),
    ProviderSpec("Zettle", ["zettle", "izettle", "i zettle"], "https://developer.zettle.com/docs/api/finance/user-guides/fetch-account-transactions"),
    ProviderSpec("Square", ["square"], "https://developer.squareup.com/docs"),
    ProviderSpec("Shopify Payments", ["shopify", "shopify payments"], "https://shopify.dev/docs/api"),
    ProviderSpec("PayPal", ["paypal", "pp *"], "https://developer.paypal.com/api/rest/"),
    ProviderSpec("GoCardless", ["gocardless", "go cardless"], "https://developer.gocardless.com/api-reference"),
    ProviderSpec("Klarna", ["klarna"], "https://docs.klarna.com/"),
    ProviderSpec("Worldpay", ["worldpay"], "https://developer.worldpay.com/products/statements"),
    ProviderSpec("Adyen", ["adyen"], "https://docs.adyen.com/reporting/settlement-reconciliation/transaction-level/settlement-details-report"),
    ProviderSpec("Barclaycard", ["barclaycard", "bcard"], "https://developer.barclays.com/"),
    ProviderSpec("Elavon", ["elavon"], "https://www.elavon.co.uk/"),
    ProviderSpec("EVO Payments", ["evo payments", "evo"], "https://evopayments.com/"),
    ProviderSpec("Teya", ["teya", "teya solutions"], "https://www.teya.com/"),
    ProviderSpec("Dojo", ["dojo", "paymentsense"], "https://docs.dojo.tech/api"),
    ProviderSpec(
        "DNA Payments",
        ["dna payments", "dnapayments", "dna payments limited", "portal.dnapayments"],
        "https://portal.dnapayments.com/",
        notes="Merchant Processing Statement PDF (processed volume + transactional fees).",
    ),
    ProviderSpec("Global Payments", ["global payments", "globalpay", "gp"], "https://developer.globalpay.com/"),
    ProviderSpec("Trust Payments", ["trust payments", "trustpayments"], "https://www.trustpayments.com/"),
    ProviderSpec("Checkout.com", ["checkout.com", "checkout com", "cko"], "https://www.checkout.com/docs"),
    ProviderSpec("Verifone", ["verifone"], "https://www.verifone.com/en/global/developers"),
    ProviderSpec("Ingenico", ["ingenico"], "https://developer.ingenico.com/"),
    ProviderSpec("NMI", ["nmi"], "https://docs.nmi.com/"),
    ProviderSpec("PayPoint", ["paypoint"], "https://www.paypoint.com/"),
    ProviderSpec("Payzone", ["payzone"], "https://www.payzone.co.uk/"),
    ProviderSpec("myPOS", ["mypos", "my pos"], "https://developers.mypos.com/"),
    ProviderSpec("Moneris", ["moneris"], "https://developer.moneris.com/"),
    ProviderSpec("PaymentSense", ["paymentsense", "payment sense"], "https://www.paysafe.com/en/products/paymentsense/"),
    ProviderSpec("Takepayments", ["take payments", "takepayments"], "https://www.takepayments.com/"),
    ProviderSpec("Handepay", ["handepay"], "https://www.handepay.co.uk/"),
    ProviderSpec("Valitor", ["valitor"], "https://www.valitor.com/"),
    ProviderSpec("Revolut Business", ["revolut"], "https://developer.revolut.com/docs/business"),
    ProviderSpec("Capital on Tap", ["capital on tap"], "https://www.capitalontap.com/en/"),
]


def provider_catalog() -> List[Dict[str, Optional[str]]]:
    """Return provider list with docs references for UI/export."""
    native = set(providers_with_native_profiles())
    return [
        {
            "provider": p.name,
            "docs_url": p.docs_url,
            "notes": p.notes,
            "parser_support": "Native profile" if p.name in native else "Generic (template/heuristic)",
        }
        for p in PROVIDER_SPECS
    ]


def detect_providers_in_text(text: str) -> List[str]:
    """Detect provider names mentioned in one narration/text field."""
    if not text:
        return []
    haystack = str(text).lower()
    hits: List[str] = []
    for spec in PROVIDER_SPECS:
        for alias in spec.aliases:
            pattern = rf"\b{re.escape(alias.lower())}\b"
            if re.search(pattern, haystack):
                hits.append(spec.name)
                break
    return sorted(set(hits))


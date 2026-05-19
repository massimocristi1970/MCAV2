# Open Banking Categorisation Audit

Source zip: `C:\Users\Massimo Cristi\OneDrive - Savvy Loan Products Ltd\Merchant Cash Advance (MCA)\Scorecard\Scorecard Development\JsonExport.zip`
Created: 2026-05-19T18:52:37

## Summary
- JSON files found: 812
- Files processed successfully: 810
- Files with errors: 0
- Transactions processed: 877,474
- Categorised revenue total: £47,247,766.33
- Uncategorised transactions: 166
- Exception rows flagged for review: 63,504

## Top Categories
- Expenses: 343,908
- Transfer Out: 219,256
- Transfer In: 173,386
- Income: 60,286
- Debt Repayments: 47,011
- Special Inflow: 16,588
- Bank Charge: 8,732
- Loans: 5,854
- Failed Payment: 2,042
- Funding Inflow: 245
- Uncategorised: 166

## Top Exception Reasons
- possible_revenue_not_counted: 41,997
- high_value_non_revenue_credit: 21,902
- small_credit_marked_loan: 1,883
- income_contains_non_revenue_keyword: 1,437
- possible_debt_repayment_marked_expense: 270
- uncategorised: 166
- failed_payment_contains_transfer: 4

## Output Files
- file_summary: `analysis_outputs\categorisation_audit_20260519\file_level_quality_summary.csv`
- exceptions: `analysis_outputs\categorisation_audit_20260519\categorisation_exceptions.csv`
- category_summary: `analysis_outputs\categorisation_audit_20260519\category_summary.csv`
- missing_fields: `analysis_outputs\categorisation_audit_20260519\missing_fields_summary.csv`
- uncategorised_patterns: `analysis_outputs\categorisation_audit_20260519\uncategorised_patterns.csv`
- plaid_vs_ours: `analysis_outputs\categorisation_audit_20260519\plaid_vs_ours_summary.csv`
- errors: `analysis_outputs\categorisation_audit_20260519\files_with_errors.csv`
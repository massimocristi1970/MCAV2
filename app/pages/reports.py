# app/pages/reports.py
"""Report generation and export functions for business finance dashboard."""

import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional


class DashboardExporter:
    """Export dashboard data to various formats."""
    
    def __init__(self):
        self.export_timestamp = datetime.now()
    
    def export_dashboard_data(
        self, 
        company_name: str,
        params: Dict[str, Any],
        metrics: Dict[str, Any], 
        scores: Dict[str, Any],
        analysis_period: str,
        revenue_insights: Dict[str, Any],
        loans_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Prepare all dashboard data for export.
        
        Args:
            company_name: Name of the company
            params: Business parameters dictionary
            metrics: Financial metrics dictionary
            scores: Scoring results dictionary
            analysis_period: Period of analysis
            revenue_insights: Revenue insights dictionary
            loans_analysis: Optional loans analysis dictionary
            
        Returns:
            Export-ready dictionary
        """
        # Clean metrics for JSON serialization (remove DataFrames)
        clean_metrics = {k: v for k, v in metrics.items() if not isinstance(v, pd.DataFrame)}
        
        export_data = {
            'export_info': {
                'company_name': company_name,
                'export_timestamp': self.export_timestamp.isoformat(),
                'analysis_period': analysis_period,
                'generated_by': 'Business Finance Scorecard v2.0'
            },
            'business_parameters': {
                'industry': params.get('industry'),
                'requested_loan': params.get('requested_loan'),
                'directors_score': params.get('directors_score'),
                'company_age_months': params.get('company_age_months'),
                'risk_factors': {
                    'business_ccj': params.get('business_ccj', False),
                    'director_ccj': params.get('director_ccj', False),
                    'poor_or_no_online_presence': params.get('poor_or_no_online_presence', False),
                    'uses_generic_email': params.get('uses_generic_email', False)
                }
            },
            'financial_metrics': clean_metrics,
            'scoring_results': {
                'subprime_score': scores.get('subprime_score'),
                'subprime_tier': scores.get('subprime_tier'),
                'subprime_recommendation': scores.get('subprime_recommendation'),
                'v2_weighted_score': scores.get('weighted_score'),
                'ml_score': scores.get('ml_score'),
                'adjusted_ml_score': scores.get('adjusted_ml_score'),
                'industry_score': scores.get('industry_score'),
                'loan_risk': scores.get('loan_risk')
            },
            'revenue_insights': {k: v for k, v in revenue_insights.items() if not isinstance(v, pd.DataFrame)},
            'loans_analysis': self._clean_loans_analysis(loans_analysis) if loans_analysis else {}
        }
        
        return export_data
    
    def _clean_loans_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Remove DataFrames from loans analysis for JSON serialization."""
        if not analysis:
            return {}
        
        clean = {}
        for k, v in analysis.items():
            if isinstance(v, pd.DataFrame):
                continue  # Skip DataFrames
            clean[k] = v
        return clean
    
    def generate_html_report(self, export_data: Dict[str, Any]) -> str:
        """
        Generate comprehensive HTML report.
        
        Args:
            export_data: Export data dictionary from export_dashboard_data()
            
        Returns:
            HTML string
        """
        def get_score_class(score):
            if score is None:
                return "low"
            if score >= 70:
                return "high"
            elif score >= 40:
                return "medium"
            else:
                return "low"
        
        # Generate loans section HTML if data exists
        loans_section = ""
        if export_data['loans_analysis'] and export_data['loans_analysis'].get('loan_count', 0) > 0:
            loans_section = f"""
            <div class="section">
                <h2>💰 Loans & Debt Analysis</h2>
                <div class="metric-grid">
                    <div class="metric-card">
                        <h4>Total Loans Received</h4>
                        <div>£{export_data['loans_analysis'].get('total_loans_received', 0):,.0f}</div>
                        <p>{export_data['loans_analysis'].get('loan_count', 0)} transactions</p>
                    </div>
                    <div class="metric-card">
                        <h4>Total Repayments</h4>
                        <div>£{export_data['loans_analysis'].get('total_repayments_made', 0):,.0f}</div>
                        <p>{export_data['loans_analysis'].get('repayment_count', 0)} transactions</p>
                    </div>
                    <div class="metric-card">
                        <h4>Net Borrowing</h4>
                        <div>£{export_data['loans_analysis'].get('net_borrowing', 0):,.0f}</div>
                    </div>
                    <div class="metric-card">
                        <h4>Repayment Ratio</h4>
                        <div>{export_data['loans_analysis'].get('repayment_ratio', 0)*100:.1f}%</div>
                    </div>
                </div>
            </div>
            """
        
        # Safe get for scores with defaults
        subprime_score = export_data['scoring_results'].get('subprime_score') or 0
        weighted_score = export_data['scoring_results'].get('v2_weighted_score') or 0
        adjusted_ml_score = export_data['scoring_results'].get('adjusted_ml_score') or 0
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Business Finance Scorecard Report - {export_data['export_info']['company_name']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
                .section {{ margin-bottom: 30px; padding: 15px; border-left: 4px solid #007bff; }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
                .metric-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }}
                .score-high {{ color: #28a745; font-weight: bold; font-size: 1.5em; }}
                .score-medium {{ color: #ffc107; font-weight: bold; font-size: 1.5em; }}
                .score-low {{ color: #dc3545; font-weight: bold; font-size: 1.5em; }}
                .table-responsive {{ overflow-x: auto; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f8f9fa; font-weight: bold; }}
                .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #6c757d; }}
            </style>
        </head>
        <body>
            <!-- Header Section -->
            <div class="header">
                <h1>🏦 Business Finance Scorecard Report</h1>
                <h2>{export_data['export_info']['company_name']}</h2>
                <p><strong>Generated:</strong> {datetime.fromisoformat(export_data['export_info']['export_timestamp']).strftime('%B %d, %Y at %I:%M %p')}</p>
                <p><strong>Analysis Period:</strong> {export_data['export_info']['analysis_period']}</p>
                <p><strong>Industry:</strong> {export_data['business_parameters']['industry']}</p>
            </div>
            
            <!-- Executive Summary -->
            <div class="section">
                <h2>📊 Executive Summary</h2>
                <div class="metric-grid">
                    <div class="metric-card">
                        <h3>🎯 Subprime Score</h3>
                        <div class="score-{get_score_class(subprime_score)}">{subprime_score:.1f}/100</div>
                        <p>{export_data['scoring_results'].get('subprime_tier', 'N/A')}</p>
                    </div>
                    <div class="metric-card">
                        <h3>🏛️ V2 Weighted</h3>
                        <div class="score-{get_score_class(weighted_score)}">{weighted_score:.0f}/100</div>
                    </div>
                    <div class="metric-card">
                        <h3>🤖 ML Score</h3>
                        <div class="score-{get_score_class(adjusted_ml_score)}">{adjusted_ml_score:.1f}%</div>
                    </div>
                    <div class="metric-card">
                        <h3>💰 Requested Loan</h3>
                        <div>£{export_data['business_parameters']['requested_loan']:,.0f}</div>
                        <p>{export_data['scoring_results'].get('loan_risk', 'N/A')}</p>
                    </div>
                </div>
            </div>
            
            <!-- Financial Metrics -->
            <div class="section">
                <h2>📈 Financial Metrics</h2>
                <div class="metric-grid">
                    <div class="metric-card">
                        <h4>Total Revenue</h4>
                        <div>£{export_data['financial_metrics'].get('Total Revenue', 0):,.0f}</div>
                    </div>
                    <div class="metric-card">
                        <h4>Total Expenses</h4>
                        <div>£{export_data['financial_metrics'].get('Total Expenses', 0):,.0f}</div>
                    </div>
                    <div class="metric-card">
                        <h4>Net Income</h4>
                        <div>£{export_data['financial_metrics'].get('Net Income', 0):,.0f}</div>
                    </div>
                    <div class="metric-card">
                        <h4>DSCR</h4>
                        <div>{export_data['financial_metrics'].get('Debt Service Coverage Ratio', 0):.2f}</div>
                    </div>
                    <div class="metric-card">
                        <h4>Operating Margin</h4>
                        <div>{export_data['financial_metrics'].get('Operating Margin', 0)*100:.1f}%</div>
                    </div>
                    <div class="metric-card">
                        <h4>Cash Flow Volatility</h4>
                        <div>{export_data['financial_metrics'].get('Cash Flow Volatility', 0):.3f}</div>
                    </div>
                </div>
            </div>
            
            <!-- Revenue Insights -->
            <div class="section">
                <h2>💵 Revenue Insights</h2>
                <div class="metric-grid">
                    <div class="metric-card">
                        <h4>Revenue Sources</h4>
                        <div>{export_data['revenue_insights'].get('unique_revenue_sources', 0)}</div>
                    </div>
                    <div class="metric-card">
                        <h4>Avg Daily Revenue</h4>
                        <div>£{export_data['revenue_insights'].get('avg_daily_revenue_amount', 0):,.0f}</div>
                    </div>
                    <div class="metric-card">
                        <h4>Revenue Days</h4>
                        <div>{export_data['revenue_insights'].get('total_revenue_days', 0)}</div>
                    </div>
                </div>
            </div>
            
            {loans_section}
            
            <!-- Business Parameters -->
            <div class="section">
                <h2>⚙️ Business Parameters</h2>
                <table>
                    <tr><th>Parameter</th><th>Value</th></tr>
                    <tr><td>Industry</td><td>{export_data['business_parameters']['industry']}</td></tr>
                    <tr><td>Company Age</td><td>{export_data['business_parameters']['company_age_months']} months</td></tr>
                    <tr><td>Directors Score</td><td>{export_data['business_parameters']['directors_score']}/100</td></tr>
                    <tr><td>Business CCJs</td><td>{'Yes' if export_data['business_parameters']['risk_factors']['business_ccj'] else 'No'}</td></tr>
                    <tr><td>Director CCJs</td><td>{'Yes' if export_data['business_parameters']['risk_factors']['director_ccj'] else 'No'}</td></tr>
                </table>
            </div>
            
            <!-- Recommendation -->
            <div class="section">
                <h2>📋 Recommendation</h2>
                <p><strong>{export_data['scoring_results'].get('subprime_recommendation', 'No recommendation available')}</strong></p>
            </div>
            
            <div class="footer">
                <p>Generated by Business Finance Scorecard v2.0</p>
                <p>This report is for informational purposes only and should not be considered as financial advice.</p>
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def generate_json_export(self, export_data: Dict[str, Any]) -> str:
        """
        Generate JSON export of dashboard data.
        
        Args:
            export_data: Export data dictionary
            
        Returns:
            JSON string
        """
        return json.dumps(export_data, indent=2, default=str)
    
    def generate_csv_metrics(self, metrics: Dict[str, Any], company_name: str) -> str:
        """
        Generate CSV export of financial metrics.
        
        Args:
            metrics: Financial metrics dictionary
            company_name: Company name for the export
            
        Returns:
            CSV string
        """
        # Filter out non-numeric metrics and DataFrames
        metric_rows = []
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                metric_rows.append({'Metric': key, 'Value': value})
        
        df = pd.DataFrame(metric_rows)
        df.insert(0, 'Company', company_name)
        df.insert(1, 'Export Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        return df.to_csv(index=False)


def get_score_summary_text(scores: Dict[str, Any]) -> str:
    """
    Generate a text summary of scores for display.
    
    Args:
        scores: Dictionary of scoring results
        
    Returns:
        Summary text string
    """
    subprime = scores.get('subprime_score', 0)
    weighted = scores.get('weighted_score', 0)
    ml = scores.get('adjusted_ml_score', scores.get('ml_score', 0)) or 0
    
    summary_lines = [
        f"**Subprime Score:** {subprime:.1f}/100",
        f"**V2 Weighted:** {weighted:.0f}/100",
        f"**ML Probability:** {ml:.1f}%",
    ]
    
    tier = scores.get('subprime_tier', 'Unknown')
    recommendation = scores.get('subprime_recommendation', 'No recommendation')
    
    summary_lines.extend([
        f"**Risk Tier:** {tier}",
        f"**Recommendation:** {recommendation}"
    ])
    
    return "\n".join(summary_lines)


def format_metrics_for_display(metrics: Dict[str, Any]) -> Dict[str, str]:
    """
    Format metrics dictionary for display purposes.
    
    Args:
        metrics: Raw metrics dictionary
        
    Returns:
        Dictionary with formatted string values
    """
    formatted = {}
    
    currency_metrics = ['Total Revenue', 'Total Expenses', 'Net Income', 
                       'Total Debt', 'Total Debt Repayments', 'Average Month-End Balance',
                       'Gross Burn Rate', 'Monthly Average Revenue']
    
    percentage_metrics = ['Operating Margin', 'Revenue Growth Rate', 
                         'Debt-to-Income Ratio', 'Expense-to-Revenue Ratio']
    
    ratio_metrics = ['Debt Service Coverage Ratio', 'Cash Flow Volatility']
    
    for key, value in metrics.items():
        if isinstance(value, pd.DataFrame):
            continue
        
        if key in currency_metrics:
            formatted[key] = f"£{value:,.0f}"
        elif key in percentage_metrics:
            formatted[key] = f"{value*100:.1f}%"
        elif key in ratio_metrics:
            formatted[key] = f"{value:.2f}"
        elif isinstance(value, float):
            formatted[key] = f"{value:.2f}"
        elif isinstance(value, int):
            formatted[key] = str(value)
        else:
            formatted[key] = str(value)
    
    return formatted

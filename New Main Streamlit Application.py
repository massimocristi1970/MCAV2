# app/main.py
"""
Enhanced Business Finance Application - Main Entry Point
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, date
import traceback

# Configure Streamlit page
st.set_page_config(
    page_title="Business Finance Scorecard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import our modules
try:
    from .config.settings import settings
    from .core.logger import get_logger, audit_logger
    from .core.exceptions import BusinessFinanceException
    from .core.cache import session_manager
    from .services.data_processor import data_processor
    from .services.financial_analyzer import financial_analyzer
    from .models.ml_predictor import ml_predictor
    from .utils.chart_utils import chart_generator, display_chart, create_metric_cards
    from .pages.overview import render_overview_page
    from .pages.bank_account import render_bank_account_page
    from .pages.upload import render_upload_page
    from .pages.dashboard import render_dashboard_page
    from .components.alerts import show_error, show_success, show_warning, show_info
    
    logger = get_logger("main")
    
except ImportError as e:
    st.error(f"Failed to import required modules: {str(e)}")
    st.stop()

class BusinessFinanceApp:
    """Main application class."""
    
    def __init__(self):
        self.logger = get_logger("app")
        
        # Initialize session state
        self._initialize_session_state()
        
        # Set up the UI
        self._setup_ui()
    
    def _initialize_session_state(self):
        """Initialize session state variables."""
        
        # Application state
        session_manager.get_or_create("app_initialized", lambda: True)
        session_manager.get_or_create("current_data", lambda: None)
        session_manager.get_or_create("current_metrics", lambda: None)
        session_manager.get_or_create("analysis_results", lambda: {})
        
        # User preferences
        session_manager.get_or_create("theme", lambda: "light")
        session_manager.get_or_create("auto_refresh", lambda: False)
        
        # Cache keys for data management
        session_manager.get_or_create("data_cache_key", lambda: None)
        
        self.logger.info("Session state initialized")
    
    def _setup_ui(self):
        """Set up the main UI structure."""
        
        # Custom CSS
        self._apply_custom_css()
        
        # Header
        self._render_header()
        
        # Sidebar
        self._render_sidebar()
        
        # Main content area
        self._render_main_content()
        
        # Footer
        self._render_footer()
    
    def _apply_custom_css(self):
        """Apply custom CSS styling."""
        
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
        }
        
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-good { background-color: #28a745; }
        .status-warning { background-color: #ffc107; }
        .status-danger { background-color: #dc3545; }
        
        .sidebar-section {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        
        .data-quality-indicator {
            font-size: 0.9em;
            padding: 0.5rem;
            border-radius: 4px;
            margin: 0.5rem 0;
        }
        
        .quality-excellent { background-color: #d4edda; color: #155724; }
        .quality-good { background-color: #fff3cd; color: #856404; }
        .quality-poor { background-color: #f8d7da; color: #721c24; }
        
        .loading-spinner {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 200px;
        }
        
        .error-container {
            background-color: #f8d7da;
            color: #721c24;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #f5c6cb;
            margin: 1rem 0;
        }
        
        .success-container {
            background-color: #d4edda;
            color: #155724;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #c3e6cb;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _render_header(self):
        """Render the application header."""
        
        st.markdown("""
        <div class="main-header">
            <h1>🏦 Business Finance Scorecard</h1>
            <p>Advanced Financial Analysis & Risk Assessment Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick status indicators
        self._render_status_indicators()
    
    def _render_status_indicators(self):
        """Render quick status indicators."""
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            data_status = "🟢 Ready" if session_manager.get("current_data") is not None else "🔴 No Data"
            st.markdown(f"**Data Status:** {data_status}")
        
        with col2:
            ml_status = "🟢 Ready" if ml_predictor.model else "🔴 Not Loaded"
            st.markdown(f"**ML Model:** {ml_status}")
        
        with col3:
            cache_status = "🟢 Active" if session_manager.get("data_cache_key") else "🔴 No Cache"
            st.markdown(f"**Cache:** {cache_status}")
        
        with col4:
            api_status = "🟢 Connected" if settings.PLAID_CLIENT_ID else "🔴 Not Configured"
            st.markdown(f"**API:** {api_status}")
    
    def _render_sidebar(self):
        """Render the sidebar with navigation and controls."""
        
        with st.sidebar:
            st.markdown("## 🧭 Navigation")
            
            # Page selection
            page = st.selectbox(
                "Select Page",
                ["📊 Executive Dashboard", "🔍 Overview & Analysis", "🏦 Bank Account", "📤 Upload Data"],
                index=0
            )
            
            session_manager.set("current_page", page)
            
            st.markdown("---")
            
            # Quick actions
            st.markdown("## ⚡ Quick Actions")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Refresh Data", help="Refresh all cached data"):
                    self._refresh_data()
            
            with col2:
                if st.button("📊 New Analysis", help="Start fresh analysis"):
                    self._reset_analysis()
            
            # Data quality indicator
            self._render_data_quality_sidebar()
            
            # Settings
            self._render_settings_sidebar()
            
            # Help section
            self._render_help_sidebar()
    
    def _render_data_quality_sidebar(self):
        """Render data quality information in sidebar."""
        
        st.markdown("---")
        st.markdown("## 📈 Data Quality")
        
        current_data = session_manager.get("current_data")
        
        if current_data is not None:
            try:
                # Get data quality report
                quality_report = data_processor.get_data_quality_report(current_data)
                
                # Display quality score
                completeness_score = quality_report.get('data_completeness_score', 0)
                
                if completeness_score >= 90:
                    quality_class = "quality-excellent"
                    quality_icon = "🟢"
                elif completeness_score >= 70:
                    quality_class = "quality-good"
                    quality_icon = "🟡"
                else:
                    quality_class = "quality-poor"
                    quality_icon = "🔴"
                
                st.markdown(f"""
                <div class="data-quality-indicator {quality_class}">
                    {quality_icon} Data Quality: {completeness_score:.1f}%
                </div>
                """, unsafe_allow_html=True)
                
                # Show key metrics
                st.markdown(f"**Transactions:** {quality_report.get('total_transactions', 0):,}")
                
                date_range = quality_report.get('date_range', {})
                if date_range.get('start_date') and date_range.get('end_date'):
                    st.markdown(f"**Period:** {date_range['days_covered']} days")
                
                # Show anomalies if any
                anomalies = quality_report.get('anomalies', [])
                if anomalies:
                    st.markdown("**⚠️ Anomalies Detected:**")
                    for anomaly in anomalies[:3]:  # Show first 3
                        st.markdown(f"• {anomaly}")
                
            except Exception as e:
                st.markdown("❌ Unable to assess data quality")
                self.logger.warning(f"Error assessing data quality: {str(e)}")
        else:
            st.markdown("📤 No data loaded")
    
    def _render_settings_sidebar(self):
        """Render settings in sidebar."""
        
        st.markdown("---")
        st.markdown("## ⚙️ Settings")
        
        # Theme selection
        theme = st.selectbox(
            "Theme",
            ["Light", "Dark", "Auto"],
            index=0
        )
        session_manager.set("theme", theme.lower())
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox(
            "Auto-refresh data",
            value=session_manager.get("auto_refresh", False),
            help="Automatically refresh data every 5 minutes"
        )
        session_manager.set("auto_refresh", auto_refresh)
        
        # Cache management
        if st.button("🗑️ Clear Cache", help="Clear all cached data"):
            self._clear_cache()
    
    def _render_help_sidebar(self):
        """Render help section in sidebar."""
        
        st.markdown("---")
        st.markdown("## ❓ Help & Support")
        
        with st.expander("📚 Quick Guide"):
            st.markdown("""
            1. **Upload Data**: Use the Upload tab to import transaction files
            2. **View Analysis**: Check the Overview tab for financial metrics
            3. **Bank Integration**: Connect to Plaid API for live data
            4. **Dashboard**: Executive summary with key insights
            """)
        
        with st.expander("🛠️ Troubleshooting"):
            st.markdown("""
            - **No data showing**: Check file format and upload again
            - **Slow performance**: Clear cache and refresh
            - **API errors**: Verify credentials in settings
            - **Chart issues**: Try refreshing the page
            """)
        
        st.markdown("---")
        st.markdown(f"**Version:** {settings.APP_VERSION}")
        st.markdown(f"**Debug Mode:** {'On' if settings.DEBUG else 'Off'}")
    
    def _render_main_content(self):
        """Render the main content area based on selected page."""
        
        current_page = session_manager.get("current_page", "📊 Executive Dashboard")
        
        try:
            # Show loading spinner for data-heavy operations
            if session_manager.get("loading", False):
                self._show_loading_spinner()
                return
            
            # Route to appropriate page
            if current_page == "📊 Executive Dashboard":
                self._render_dashboard_page()
            elif current_page == "🔍 Overview & Analysis":
                self._render_overview_page()
            elif current_page == "🏦 Bank Account":
                self._render_bank_account_page()
            elif current_page == "📤 Upload Data":
                self._render_upload_page()
            else:
                st.error(f"Unknown page: {current_page}")
        
        except BusinessFinanceException as e:
            self._handle_business_error(e)
        except Exception as e:
            self._handle_unexpected_error(e)
    
    def _render_dashboard_page(self):
        """Render the executive dashboard page."""
        
        st.markdown("## 📊 Executive Dashboard")
        st.markdown("---")
        
        current_data = session_manager.get("current_data")
        current_metrics = session_manager.get("current_metrics")
        
        if current_data is None:
            self._show_no_data_message()
            return
        
        # Key metrics cards
        if current_metrics:
            st.markdown("### 📈 Key Performance Indicators")
            create_metric_cards(current_metrics)
            st.markdown("---")
        
        # Main dashboard charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💰 Revenue & Expenses")
            try:
                revenue_chart = chart_generator.create_revenue_expense_chart(current_data)
                display_chart(revenue_chart)
            except Exception as e:
                st.error(f"Error creating revenue chart: {str(e)}")
        
        with col2:
            st.markdown("#### 💧 Cash Flow Analysis")
            try:
                cashflow_chart = chart_generator.create_cash_flow_chart(current_data)
                display_chart(cashflow_chart)
            except Exception as e:
                st.error(f"Error creating cash flow chart: {str(e)}")
        
        # Additional insights
        if current_metrics:
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown("#### 🎯 Risk Assessment")
                try:
                    risk_chart = chart_generator.create_risk_assessment_chart(
                        current_metrics, {}  # TODO: Add industry benchmarks
                    )
                    display_chart(risk_chart)
                except Exception as e:
                    st.error(f"Error creating risk chart: {str(e)}")
            
            with col4:
                st.markdown("#### 📊 Transaction Categories")
                try:
                    category_chart = chart_generator.create_transaction_category_chart(current_data)
                    display_chart(category_chart)
                except Exception as e:
                    st.error(f"Error creating category chart: {str(e)}")
        
        # Financial summary
        if current_metrics:
            self._render_financial_summary(current_metrics)
    
    def _render_overview_page(self):
        """Render the overview and analysis page."""
        
        st.markdown("## 🔍 Overview & Analysis")
        st.markdown("---")
        
        # Business information inputs
        self._render_business_inputs()
        
        current_data = session_manager.get("current_data")
        
        if current_data is not None:
            # Analysis tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Current Period", "📅 Last 3 Months", "📆 Last 6 Months", "📈 Trends & Analysis"])
            
            with tab1:
                self._render_current_analysis(current_data)
            
            with tab2:
                filtered_data = data_processor.filter_data_by_timeframe(current_data, 3)
                self._render_period_analysis(filtered_data, "3 Months")
            
            with tab3:
                filtered_data = data_processor.filter_data_by_timeframe(current_data, 6)
                self._render_period_analysis(filtered_data, "6 Months")
            
            with tab4:
                self._render_trend_analysis(current_data)
        else:
            self._show_no_data_message()
    
    def _render_bank_account_page(self):
        """Render the bank account page."""
        
        st.markdown("## 🏦 Bank Account Information")
        st.markdown("---")
        
        # Data source selection
        data_source = st.radio(
            "Choose data source:",
            ["🔗 Plaid API", "📁 Upload File"],
            horizontal=True
        )
        
        if data_source == "📁 Upload File":
            self._render_file_upload_section()
        else:
            self._render_plaid_integration_section()
    
    def _render_upload_page(self):
        """Render the data upload page."""
        
        st.markdown("## 📤 Upload Transaction Data")
        st.markdown("---")
        
        # File upload section
        self._render_file_upload_interface()
    
    def _render_business_inputs(self):
        """Render business information input form."""
        
        with st.expander("🏢 Business Information", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                loan_amount = st.number_input(
                    "Requested Loan Amount (£)",
                    min_value=0.0,
                    value=session_manager.get("loan_amount", 50000.0),
                    step=1000.0,
                    format="%.2f"
                )
                session_manager.set("loan_amount", loan_amount)
            
            with col2:
                from .config.industry_config import INDUSTRY_THRESHOLDS
                industry = st.selectbox(
                    "Industry Sector",
                    list(INDUSTRY_THRESHOLDS.keys()),
                    index=0
                )
                session_manager.set("industry", industry)
            
            with col3:
                company_age = st.number_input(
                    "Company Age (months)",
                    min_value=0,
                    max_value=1200,
                    value=session_manager.get("company_age", 24),
                    step=1
                )
                session_manager.set("company_age", company_age)
            
            # Director and credit information
            col4, col5 = st.columns(2)
            
            with col4:
                director_score = st.number_input(
                    "Director Credit Score",
                    min_value=0,
                    max_value=100,
                    value=session_manager.get("director_score", 75),
                    step=1
                )
                session_manager.set("director_score", director_score)
            
            with col5:
                st.markdown("**Credit History Flags**")
                personal_default = st.checkbox("Personal defaults (12m)", value=False)
                business_ccj = st.checkbox("Business CCJs", value=False)
                director_ccj = st.checkbox("Director CCJs", value=False)
                
                session_manager.set("credit_flags", {
                    "personal_default": personal_default,
                    "business_ccj": business_ccj,
                    "director_ccj": director_ccj
                })
            
            # Digital footprint
            st.markdown("**Digital Footprint Assessment**")
            col6, col7 = st.columns(2)
            
            with col6:
                outdated_web = st.checkbox("Website/social outdated (3m+)", value=False)
                generic_email = st.checkbox("Generic email provider", value=False)
            
            with col7:
                no_web_presence = st.checkbox("No online presence", value=False)
            
            session_manager.set("digital_flags", {
                "outdated_web": outdated_web,
                "generic_email": generic_email,
                "no_web_presence": no_web_presence
            })
    
    def _render_current_analysis(self, data: pd.DataFrame):
        """Render analysis for current period."""
        
        try:
            # Calculate comprehensive metrics
            with st.spinner("Calculating financial metrics..."):
                company_age = session_manager.get("company_age", 24)
                metrics = financial_analyzer.calculate_comprehensive_metrics(
                    data, company_age, include_advanced=True
                )
                session_manager.set("current_metrics", metrics)
            
            # Display metrics
            st.markdown("### 📊 Financial Metrics")
            
            # Basic metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Revenue",
                    f"£{metrics['Total Revenue']:,.2f}",
                    delta=f"{metrics.get('Revenue Growth Rate', 0):+.1f}%"
                )
            
            with col2:
                st.metric(
                    "Net Income",
                    f"£{metrics['Net Income']:,.2f}",
                    delta=None
                )
            
            with col3:
                st.metric(
                    "Operating Margin",
                    f"{metrics['Operating Margin']*100:.1f}%",
                    delta=None
                )
            
            with col4:
                st.metric(
                    "DSCR",
                    f"{metrics['Debt Service Coverage Ratio']:.2f}",
                    delta=None
                )
            
            # Advanced metrics table
            st.markdown("### 📈 Detailed Metrics")
            
            metrics_df = pd.DataFrame([
                {"Metric": k, "Value": v} 
                for k, v in metrics.items() 
                if isinstance(v, (int, float))
            ])
            
            st.dataframe(metrics_df, use_container_width=True)
            
            # ML Prediction
            self._render_ml_prediction(metrics)
            
            # Industry benchmarking
            self._render_industry_benchmark(metrics)
            
        except Exception as e:
            st.error(f"Error in current analysis: {str(e)}")
            self.logger.error(f"Current analysis error: {str(e)}")
    
    def _render_period_analysis(self, data: pd.DataFrame, period_name: str):
        """Render analysis for a specific time period."""
        
        if data.empty:
            st.warning(f"No data available for the last {period_name.lower()}")
            return
        
        try:
            # Calculate metrics for the period
            company_age = session_manager.get("company_age", 24)
            metrics = financial_analyzer.calculate_comprehensive_metrics(
                data, company_age, include_advanced=False
            )
            
            st.markdown(f"### 📊 {period_name} Analysis")
            
            # Key metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Revenue", f"£{metrics['Total Revenue']:,.2f}")
                st.metric("Expenses", f"£{metrics['Total Expenses']:,.2f}")
            
            with col2:
                st.metric("Net Income", f"£{metrics['Net Income']:,.2f}")
                st.metric("Operating Margin", f"{metrics['Operating Margin']*100:.1f}%")
            
            with col3:
                st.metric("DSCR", f"{metrics['Debt Service Coverage Ratio']:.2f}")
                st.metric("Cash Flow Volatility", f"{metrics['Cash Flow Volatility']:.3f}")
            
            # Period-specific charts
            revenue_chart = chart_generator.create_revenue_expense_chart(data)
            display_chart(revenue_chart)
            
        except Exception as e:
            st.error(f"Error in {period_name.lower()} analysis: {str(e)}")
    
    def _render_ml_prediction(self, metrics: Dict):
        """Render ML prediction results."""
        
        st.markdown("### 🤖 AI Risk Assessment")
        
        try:
            # Get prediction parameters
            director_score = session_manager.get("director_score", 75)
            company_age = session_manager.get("company_age", 24)
            industry = session_manager.get("industry", "Other")
            from .config.industry_config import INDUSTRY_THRESHOLDS
            sector_risk = INDUSTRY_THRESHOLDS[industry].get("Sector Risk", 1)
            
            # Get ML prediction
            with st.spinner("Running AI analysis..."):
                prediction = ml_predictor.predict_repayment_probability(
                    metrics, director_score, sector_risk, company_age, include_confidence=True
                )
            
            # Display prediction results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Repayment Probability",
                    f"{prediction['probability']:.1f}%",
                    delta=None
                )
            
            with col2:
                st.metric(
                    "Risk Category",
                    prediction['risk_category'],
                    delta=None
                )
            
            with col3:
                st.metric(
                    "Model Confidence",
                    f"{prediction['model_confidence']:.1f}%",
                    delta=None
                )
            
            # Confidence interval
            if 'confidence_interval' in prediction:
                ci = prediction['confidence_interval']
                st.info(f"📊 **Confidence Interval:** {ci['lower']:.1f}% - {ci['upper']:.1f}% (width: {ci['width']:.1f}%)")
            
            # Feature importance
            if 'feature_importance' in prediction:
                st.markdown("#### 🎯 Key Factors")
                importance = prediction['feature_importance']
                
                for factor, impact in list(importance.items())[:5]:
                    st.write(f"• **{factor}**: {impact:.1f}% impact")
            
            # Explanation
            if 'explanation' in prediction:
                explanation = prediction['explanation']
                
                with st.expander("📝 Detailed Analysis"):
                    st.write(explanation['summary'])
                    
                    if explanation['key_factors']:
                        st.markdown("**Strengths:**")
                        for factor in explanation['key_factors']:
                            st.write(f"✅ {factor}")
                    
                    if explanation['risk_factors']:
                        st.markdown("**Risk Areas:**")
                        for factor in explanation['risk_factors']:
                            st.write(f"⚠️ {factor}")
                    
                    if explanation['recommendations']:
                        st.markdown("**Recommendations:**")
                        for rec in explanation['recommendations']:
                            st.write(f"💡 {rec}")
        
        except Exception as e:
            st.error(f"Error in AI analysis: {str(e)}")
            self.logger.error(f"ML prediction error: {str(e)}")
    
    def _render_industry_benchmark(self, metrics: Dict):
        """Render industry benchmarking results."""
        
        st.markdown("### 🏭 Industry Benchmarking")
        
        try:
            industry = session_manager.get("industry", "Other")
            
            # Get benchmark comparison
            benchmark_results = financial_analyzer.benchmark_against_industry(metrics, industry)
            
            if "error" in benchmark_results:
                st.warning(benchmark_results["error"])
                return
            
            # Overall score
            overall = benchmark_results.get("overall_score", {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Benchmark Score",
                    f"{overall.get('score', 0):.1f}%",
                    delta=None
                )
            
            with col2:
                st.metric(
                    "Metrics Above Benchmark",
                    f"{overall.get('metrics_above_benchmark', 0)}/{overall.get('total_metrics', 0)}",
                    delta=None
                )
            
            with col3:
                st.metric(
                    "Industry",
                    industry,
                    delta=None
                )
            
            # Detailed benchmark comparison
            benchmark_df = []
            for metric_name, result in benchmark_results.items():
                if isinstance(result, dict) and 'actual' in result:
                    benchmark_df.append({
                        'Metric': metric_name,
                        'Your Value': result['actual'],
                        'Industry Benchmark': result['benchmark'],
                        'Performance': result['performance'],
                        'Variance %': result['variance_percent']
                    })
            
            if benchmark_df:
                st.dataframe(pd.DataFrame(benchmark_df), use_container_width=True)
        
        except Exception as e:
            st.error(f"Error in benchmarking: {str(e)}")
    
    def _render_financial_summary(self, metrics: Dict):
        """Render financial summary section."""
        
        st.markdown("---")
        st.markdown("### 📋 Financial Summary")
        
        try:
            summary = financial_analyzer.generate_financial_summary(metrics)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 💪 Key Strengths")
                strengths = summary.get('key_strengths', [])
                if strengths:
                    for strength in strengths:
                        st.write(f"✅ {strength}")
                else:
                    st.write("No significant strengths identified")
            
            with col2:
                st.markdown("#### ⚠️ Risk Areas")
                risks = summary.get('risk_areas', [])
                if risks:
                    for risk in risks:
                        st.write(f"🔴 {risk}")
                else:
                    st.write("No significant risks identified")
            
            # Overall health assessment
            health = summary.get('financial_health', 'Unknown')
            health_colors = {
                'Excellent': '🟢',
                'Good': '🟡', 
                'Fair': '🟠',
                'Poor': '🔴',
                'Critical': '🔴'
            }
            
            st.markdown(f"#### 🏥 Overall Financial Health: {health_colors.get(health, '⚪')} {health}")
            
            # Recommendations
            recommendations = summary.get('recommendations', [])
            if recommendations:
                st.markdown("#### 💡 Recommendations")
                for rec in recommendations:
                    st.write(f"• {rec}")
        
        except Exception as e:
            st.error(f"Error generating summary: {str(e)}")
    
    def _render_file_upload_interface(self):
        """Render file upload interface."""
        
        st.markdown("### 📁 File Upload")
        
        # Date range selection
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=date.today() - timedelta(days=90),
                max_value=date.today()
            )
        
        with col2:
            end_date = st.date_input(
                "End Date", 
                value=date.today(),
                min_value=start_date,
                max_value=date.today()
            )
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload Transaction JSON",
            type=["json"],
            help="Upload a JSON file containing transaction data from Plaid or similar format"
        )
        
        if uploaded_file:
            try:
                with st.spinner("Processing uploaded file..."):
                    # Process the uploaded file
                    account_df, transaction_df = data_processor.process_uploaded_file(
                        uploaded_file, start_date, end_date
                    )
                    
                    # Store in session state
                    session_manager.set("current_data", transaction_df)
                    session_manager.set("account_data", account_df)
                    
                    # Generate cache key
                    import hashlib
                    cache_key = hashlib.md5(f"{uploaded_file.name}_{start_date}_{end_date}".encode()).hexdigest()
                    session_manager.set("data_cache_key", cache_key)
                
                show_success(f"✅ Successfully processed {len(transaction_df)} transactions from {len(account_df)} accounts")
                
                # Show data preview
                st.markdown("#### 👀 Data Preview")
                st.dataframe(transaction_df.head(10), use_container_width=True)
                
                # Data quality report
                quality_report = data_processor.get_data_quality_report(transaction_df)
                
                with st.expander("📊 Data Quality Report"):
                    st.json(quality_report)
            
            except Exception as e:
                show_error(f"❌ Error processing file: {str(e)}")
    
    def _show_loading_spinner(self):
        """Show loading spinner."""
        
        st.markdown("""
        <div class="loading-spinner">
            <div>🔄 Processing... Please wait</div>
        </div>
        """, unsafe_allow_html=True)
    
    def _show_no_data_message(self):
        """Show no data available message."""
        
        st.info("📤 No transaction data loaded. Please upload a file or connect to your bank account to begin analysis.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📁 Upload File", type="primary"):
                session_manager.set("current_page", "📤 Upload Data")
                st.rerun()
        
        with col2:
            if st.button("🔗 Connect Bank Account"):
                session_manager.set("current_page", "🏦 Bank Account")
                st.rerun()
    
    def _handle_business_error(self, error: BusinessFinanceException):
        """Handle business logic errors."""
        
        show_error(f"❌ {error.message}")
        
        if error.details:
            with st.expander("Error Details"):
                st.json(error.details)
    
    def _handle_unexpected_error(self, error: Exception):
        """Handle unexpected errors."""
        
        error_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        show_error(f"❌ An unexpected error occurred (ID: {error_id})")
        
        # Log the error
        self.logger.error(f"Unexpected error {error_id}: {str(error)}")
        audit_logger.log_error(f"Unexpected_{error_id}", str(error))
        
        if settings.DEBUG:
            st.exception(error)
    
    def _refresh_data(self):
        """Refresh all cached data."""
        
        try:
            from .core.cache import cache_manager
            cache_manager.clear_cache()
            
            # Clear session data
            session_manager.delete("current_data")
            session_manager.delete("current_metrics")
            session_manager.delete("data_cache_key")
            
            show_success("🔄 Data refreshed successfully")
            st.rerun()
        
        except Exception as e:
            show_error(f"❌ Error refreshing data: {str(e)}")
    
    def _reset_analysis(self):
        """Reset analysis and start fresh."""
        
        try:
            # Clear all analysis data
            session_manager.clear_session()
            
            show_success("🆕 Analysis reset. Starting fresh!")
            st.rerun()
        
        except Exception as e:
            show_error(f"❌ Error resetting analysis: {str(e)}")
    
    def _clear_cache(self):
        """Clear application cache."""
        
        try:
            from .core.cache import cache_manager
            cache_manager.clear_cache()
            
            show_success("🗑️ Cache cleared successfully")
        
        except Exception as e:
            show_error(f"❌ Error clearing cache: {str(e)}")
    
    def _render_footer(self):
        """Render application footer."""
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"**{settings.APP_NAME}** v{settings.APP_VERSION}")
        
        with col2:
            st.markdown("Built with ❤️ using Streamlit")
        
        with col3:
            st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

def main():
    """Main application entry point."""
    
    try:
        # Initialize and run the application
        app = BusinessFinanceApp()
        
        # Log application start
        logger.info("Business Finance Application started")
        audit_logger.log_data_access("system", "application", "startup")
        
    except Exception as e:
        st.error(f"❌ Failed to initialize application: {str(e)}")
        
        if settings.DEBUG:
            st.exception(e)
        
        st.stop()

if __name__ == "__main__":
    main()
# app/components/alerts.py
"""Alert and notification components for the application."""

import streamlit as st
from typing import Optional, Dict, Any
from datetime import datetime

def show_success(message: str, details: Optional[Dict[str, Any]] = None) -> None:
    """Display a success alert."""
    
    st.markdown(f"""
    <div class="success-container">
        <strong>Success</strong><br>
        {message}
    </div>
    """, unsafe_allow_html=True)
    
    if details:
        with st.expander("View Details"):
            st.json(details)

def show_error(message: str, details: Optional[Dict[str, Any]] = None) -> None:
    """Display an error alert."""
    
    st.markdown(f"""
    <div class="error-container">
        <strong>Error</strong><br>
        {message}
    </div>
    """, unsafe_allow_html=True)
    
    if details:
        with st.expander("Error Details"):
            st.json(details)

def show_warning(message: str, details: Optional[Dict[str, Any]] = None) -> None:
    """Display a warning alert."""
    
    st.warning(f"**Warning:** {message}")
    
    if details:
        with st.expander("Warning Details"):
            st.json(details)

def show_info(message: str, details: Optional[Dict[str, Any]] = None) -> None:
    """Display an info alert."""
    
    st.info(f"**Info:** {message}")
    
    if details:
        with st.expander("Additional Information"):
            st.json(details)

def show_progress_alert(message: str, progress: float) -> None:
    """Display a progress alert with progress bar."""
    
    st.info(message)
    st.progress(progress)

def show_data_quality_alert(quality_score: float, issues: list = None) -> None:
    """Display data quality alert with appropriate styling."""
    
    if quality_score >= 90:
        alert_type = "success"
        status = "Excellent"
    elif quality_score >= 70:
        alert_type = "warning"
        status = "Good"
    else:
        alert_type = "error"
        status = "Poor"

    message = f"**Data quality: {status}** ({quality_score:.1f}%)"
    
    if alert_type == "success":
        st.success(message)
    elif alert_type == "warning":
        st.warning(message)
    else:
        st.error(message)
    
    if issues:
        with st.expander("Quality Issues"):
            for issue in issues:
                st.write(f"• {issue}")

def show_financial_health_alert(health_status: str, score: float) -> None:
    """Display financial health status alert."""
    
    status_config = {
        "Excellent": {"color": "success"},
        "Good": {"color": "success"},
        "Fair": {"color": "warning"},
        "Poor": {"color": "error"},
        "Critical": {"color": "error"},
    }

    config = status_config.get(health_status, {"color": "info"})

    message = f"**Financial health: {health_status}** (score: {score:.1f}/100)"
    
    if config["color"] == "success":
        st.success(message)
    elif config["color"] == "warning":
        st.warning(message)
    elif config["color"] == "error":
        st.error(message)
    else:
        st.info(message)

def show_risk_assessment_alert(risk_level: str, probability: float) -> None:
    """Display risk assessment alert."""
    
    risk_config = {
        "Very Low Risk": {"color": "success"},
        "Low Risk": {"color": "success"},
        "Moderate Risk": {"color": "warning"},
        "High Risk": {"color": "error"},
        "Very High Risk": {"color": "error"},
    }

    config = risk_config.get(risk_level, {"color": "info"})

    message = f"**Risk level: {risk_level}** (repayment probability: {probability:.1f}%)"
    
    if config["color"] == "success":
        st.success(message)
    elif config["color"] == "warning":
        st.warning(message)
    elif config["color"] == "error":
        st.error(message)
    else:
        st.info(message)

def show_benchmark_alert(score: float, metrics_passed: int, total_metrics: int) -> None:
    """Display industry benchmark performance alert."""
    
    if score >= 80:
        alert_type = "success"
        status = "Exceeds industry standards"
    elif score >= 60:
        alert_type = "success"
        status = "Meets industry standards"
    elif score >= 40:
        alert_type = "warning"
        status = "Below industry average"
    else:
        alert_type = "error"
        status = "Well below industry standards"

    message = f"**{status}** ({metrics_passed}/{total_metrics} benchmarks met, {score:.1f}% score)"
    
    if alert_type == "success":
        st.success(message)
    elif alert_type == "warning":
        st.warning(message)
    else:
        st.error(message)

def show_loading_alert(message: str) -> None:
    """Display a loading alert with spinner."""
    
    st.info(message)

def show_ml_prediction_alert(probability: float, confidence: float) -> None:
    """Display ML prediction results alert."""

    if probability >= 80:
        alert_type = "success"
        risk_text = "Low risk"
    elif probability >= 60:
        alert_type = "success"
        risk_text = "Moderate risk"
    elif probability >= 40:
        alert_type = "warning"
        risk_text = "High risk"
    else:
        alert_type = "error"
        risk_text = "Very high risk"

    message = (
        f"**AI assessment: {risk_text}** (repayment probability: {probability:.1f}%, "
        f"model confidence: {confidence:.1f}%)"
    )
    
    if alert_type == "success":
        st.success(message)
    elif alert_type == "warning":
        st.warning(message)
    else:
        st.error(message)

def show_data_upload_success(filename: str, records: int, accounts: int) -> None:
    """Display successful data upload alert."""
    
    show_success(
        f"Successfully uploaded **{filename}**",
        details={
            "Records Processed": records,
            "Accounts Found": accounts,
            "Upload Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    )

def show_validation_error(field_name: str, error_message: str) -> None:
    """Display validation error for specific field."""
    
    st.error(f"**Validation error in {field_name}:** {error_message}")

def show_api_connection_status(service: str, is_connected: bool, details: str = None) -> None:
    """Display API connection status."""
    
    if is_connected:
        message = f"**{service} connected**"
        if details:
            message += f" — {details}"
        st.success(message)
    else:
        message = f"**{service} not connected**"
        if details:
            message += f" — {details}"
        st.error(message)

def show_feature_importance_alert(top_factors: list, positive_factors: int) -> None:
    """Display feature importance summary alert."""
    
    if positive_factors >= len(top_factors) * 0.7:  # 70% positive
        alert_type = "success"
        message = f"Strong financial position with {positive_factors}/{len(top_factors)} positive key factors"
    elif positive_factors >= len(top_factors) * 0.4:  # 40% positive
        alert_type = "warning"
        message = f"Mixed financial position with {positive_factors}/{len(top_factors)} positive key factors"
    else:
        alert_type = "error"
        message = f"Concerning financial position with only {positive_factors}/{len(top_factors)} positive key factors"

    full_message = f"**Key factors analysis:** {message}"
    
    if alert_type == "success":
        st.success(full_message)
    elif alert_type == "warning":
        st.warning(full_message)
    else:
        st.error(full_message)

def show_seasonal_pattern_alert(best_month: str, worst_month: str, volatility: float) -> None:
    """Display seasonal pattern analysis alert."""
    
    if volatility < 0.2:
        alert_type = "success"
        stability = "Stable"
    elif volatility < 0.5:
        alert_type = "warning"
        stability = "Moderate"
    else:
        alert_type = "error"
        stability = "Volatile"

    message = (
        f"**Seasonal analysis:** {stability} revenue pattern "
        f"(best: {best_month}, worst: {worst_month}, volatility: {volatility:.2f})"
    )
    
    if alert_type == "success":
        st.success(message)
    elif alert_type == "warning":
        st.warning(message)
    else:
        st.error(message)

def show_cash_flow_alert(negative_days: int, total_days: int, avg_balance: float) -> None:
    """Display cash flow analysis alert."""
    
    negative_ratio = negative_days / total_days if total_days > 0 else 0
    
    if negative_ratio < 0.1 and avg_balance > 5000:  # Less than 10% negative days and good balance
        alert_type = "success"
        status = "Healthy cash flow"
    elif negative_ratio < 0.25 or avg_balance > 1000:  # Less than 25% negative days or decent balance
        alert_type = "warning"
        status = "Moderate cash flow concerns"
    else:
        alert_type = "error"
        status = "Significant cash flow issues"

    message = (
        f"**{status}:** {negative_days}/{total_days} negative cash flow days, "
        f"£{avg_balance:,.2f} average balance"
    )
    
    if alert_type == "success":
        st.success(message)
    elif alert_type == "warning":
        st.warning(message)
    else:
        st.error(message)

def show_debt_management_alert(dscr: float, debt_ratio: float) -> None:
    """Display debt management assessment alert."""
    
    if dscr >= 1.5 and debt_ratio < 0.3:  # Good DSCR and low debt ratio
        alert_type = "success"
        status = "Excellent debt management"
    elif dscr >= 1.2 and debt_ratio < 0.5:  # Acceptable DSCR and moderate debt ratio
        alert_type = "warning"
        status = "Acceptable debt management"
    else:
        alert_type = "error"
        status = "Debt management concerns"

    message = f"**{status}:** DSCR {dscr:.2f}, debt ratio {debt_ratio*100:.1f}%"
    
    if alert_type == "success":
        st.success(message)
    elif alert_type == "warning":
        st.warning(message)
    else:
        st.error(message)

def show_transaction_anomaly_alert(anomalies: list) -> None:
    """Display transaction anomaly detection alert."""
    
    if not anomalies:
        st.success("**Transaction analysis:** No significant anomalies detected")
    else:
        severity = len(anomalies)

        if severity <= 2:
            alert_type = "warning"
            status = "Minor anomalies"
        else:
            alert_type = "error"
            status = "Multiple anomalies"

        message = f"**{status} detected:** {severity} potential issues found"
        
        if alert_type == "warning":
            st.warning(message)
        else:
            st.error(message)
        
        with st.expander("View Anomalies"):
            for anomaly in anomalies:
                st.write(f"• {anomaly}")

def show_recommendation_alert(recommendations: list, priority: str = "medium") -> None:
    """Display recommendations alert."""
    
    if not recommendations:
        st.info("**No specific recommendations at this time**")
        return

    priority_config = {
        "high": {"color": "error", "text": "Urgent recommendations"},
        "medium": {"color": "warning", "text": "Recommendations"},
        "low": {"color": "info", "text": "Suggestions"},
    }

    config = priority_config.get(priority, priority_config["medium"])

    message = f"**{config['text']}:** {len(recommendations)} action items identified"
    
    if config["color"] == "error":
        st.error(message)
    elif config["color"] == "warning":
        st.warning(message)
    else:
        st.info(message)
    
    with st.expander("View All Recommendations"):
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")

def show_export_success_alert(export_type: str, filename: str) -> None:
    """Display successful export alert."""
    
    show_success(
        f"{export_type} exported successfully as **{filename}**",
        details={
            "Export Type": export_type,
            "Filename": filename,
            "Export Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    )

def show_cache_status_alert(cache_size: int, last_refresh: datetime) -> None:
    """Display cache status information."""
    
    time_since_refresh = datetime.now() - last_refresh
    hours_since = time_since_refresh.total_seconds() / 3600
    
    if hours_since < 1:
        status = "Fresh"
    elif hours_since < 4:
        status = "Current"
    else:
        status = "Stale"

    st.info(
        f"**Cache status:** {status} ({cache_size} items, last refresh: {hours_since:.1f}h ago)"
    )

# Custom alert with custom styling
def show_custom_alert(
    message: str, 
    alert_type: str = "info", 
    icon: str = None, 
    expandable_content: Dict[str, Any] = None
) -> None:
    """Display a custom alert with flexible styling."""
    
    display_icon = (icon or "").strip()
    full_message = f"{display_icon} {message}".strip() if display_icon else message
    
    # Display based on type
    if alert_type == "success":
        st.success(full_message)
    elif alert_type == "error":
        st.error(full_message)
    elif alert_type == "warning":
        st.warning(full_message)
    else:
        st.info(full_message)
    
    # Add expandable content if provided
    if expandable_content:
        with st.expander("Details"):
            for key, value in expandable_content.items():
                if isinstance(value, dict):
                    st.json(value)
                elif isinstance(value, list):
                    for item in value:
                        st.write(f"• {item}")
                else:
                    st.write(f"**{key}:** {value}")

# Batch alert for multiple messages
def show_batch_alerts(alerts: list) -> None:
    """Display multiple alerts in sequence."""
    
    for alert in alerts:
        alert_type = alert.get("type", "info")
        message = alert.get("message", "")
        details = alert.get("details")
        
        if alert_type == "success":
            show_success(message, details)
        elif alert_type == "error":
            show_error(message, details)
        elif alert_type == "warning":
            show_warning(message, details)
        else:
            show_info(message, details)
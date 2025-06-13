# app/components/alerts.py
"""Alert and notification components for the application."""

import streamlit as st
from typing import Optional, Dict, Any
from datetime import datetime

def show_success(message: str, details: Optional[Dict[str, Any]] = None) -> None:
    """Display a success alert."""
    
    st.markdown(f"""
    <div class="success-container">
        <strong>✅ Success</strong><br>
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
        <strong>❌ Error</strong><br>
        {message}
    </div>
    """, unsafe_allow_html=True)
    
    if details:
        with st.expander("Error Details"):
            st.json(details)

def show_warning(message: str, details: Optional[Dict[str, Any]] = None) -> None:
    """Display a warning alert."""
    
    st.warning(f"⚠️ **Warning:** {message}")
    
    if details:
        with st.expander("Warning Details"):
            st.json(details)

def show_info(message: str, details: Optional[Dict[str, Any]] = None) -> None:
    """Display an info alert."""
    
    st.info(f"ℹ️ **Info:** {message}")
    
    if details:
        with st.expander("Additional Information"):
            st.json(details)

def show_progress_alert(message: str, progress: float) -> None:
    """Display a progress alert with progress bar."""
    
    st.info(f"🔄 {message}")
    st.progress(progress)

def show_data_quality_alert(quality_score: float, issues: list = None) -> None:
    """Display data quality alert with appropriate styling."""
    
    if quality_score >= 90:
        alert_type = "success"
        icon = "✅"
        status = "Excellent"
    elif quality_score >= 70:
        alert_type = "warning"
        icon = "⚠️"
        status = "Good"
    else:
        alert_type = "error"
        icon = "❌"
        status = "Poor"
    
    message = f"{icon} **Data Quality: {status}** ({quality_score:.1f}%)"
    
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
        "Excellent": {"color": "success", "icon": "🟢"},
        "Good": {"color": "success", "icon": "🟡"},
        "Fair": {"color": "warning", "icon": "🟠"},
        "Poor": {"color": "error", "icon": "🔴"},
        "Critical": {"color": "error", "icon": "🔴"}
    }
    
    config = status_config.get(health_status, {"color": "info", "icon": "⚪"})
    
    message = f"{config['icon']} **Financial Health: {health_status}** (Score: {score:.1f}/100)"
    
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
        "Very Low Risk": {"color": "success", "icon": "🟢"},
        "Low Risk": {"color": "success", "icon": "🟡"},
        "Moderate Risk": {"color": "warning", "icon": "🟠"},
        "High Risk": {"color": "error", "icon": "🔴"},
        "Very High Risk": {"color": "error", "icon": "🔴"}
    }
    
    config = risk_config.get(risk_level, {"color": "info", "icon": "⚪"})
    
    message = f"{config['icon']} **Risk Level: {risk_level}** (Repayment Probability: {probability:.1f}%)"
    
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
        icon = "🏆"
        status = "Exceeds Industry Standards"
    elif score >= 60:
        alert_type = "success"
        icon = "✅"
        status = "Meets Industry Standards"
    elif score >= 40:
        alert_type = "warning"
        icon = "⚠️"
        status = "Below Industry Average"
    else:
        alert_type = "error"
        icon = "❌"
        status = "Well Below Industry Standards"
    
    message = f"{icon} **{status}** ({metrics_passed}/{total_metrics} benchmarks met, {score:.1f}% score)"
    
    if alert_type == "success":
        st.success(message)
    elif alert_type == "warning":
        st.warning(message)
    else:
        st.error(message)

def show_loading_alert(message: str) -> None:
    """Display a loading alert with spinner."""
    
    st.info(f"🔄 {message}")

def show_ml_prediction_alert(probability: float, confidence: float) -> None:
    """Display ML prediction results alert."""
    
    if probability >= 80:
        alert_type = "success"
        icon = "🟢"
        risk_text = "Low Risk"
    elif probability >= 60:
        alert_type = "success"
        icon = "🟡"
        risk_text = "Moderate Risk"
    elif probability >= 40:
        alert_type = "warning"
        icon = "🟠"
        risk_text = "High Risk"
    else:
        alert_type = "error"
        icon = "🔴"
        risk_text = "Very High Risk"
    
    message = f"{icon} **AI Assessment: {risk_text}** (Repayment Probability: {probability:.1f}%, Model Confidence: {confidence:.1f}%)"
    
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
    
    st.error(f"❌ **Validation Error in {field_name}:** {error_message}")

def show_api_connection_status(service: str, is_connected: bool, details: str = None) -> None:
    """Display API connection status."""
    
    if is_connected:
        message = f"🟢 **{service} Connected**"
        if details:
            message += f" - {details}"
        st.success(message)
    else:
        message = f"🔴 **{service} Not Connected**"
        if details:
            message += f" - {details}"
        st.error(message)

def show_feature_importance_alert(top_factors: list, positive_factors: int) -> None:
    """Display feature importance summary alert."""
    
    if positive_factors >= len(top_factors) * 0.7:  # 70% positive
        alert_type = "success"
        icon = "✅"
        message = f"Strong financial position with {positive_factors}/{len(top_factors)} positive key factors"
    elif positive_factors >= len(top_factors) * 0.4:  # 40% positive
        alert_type = "warning" 
        icon = "⚠️"
        message = f"Mixed financial position with {positive_factors}/{len(top_factors)} positive key factors"
    else:
        alert_type = "error"
        icon = "❌"
        message = f"Concerning financial position with only {positive_factors}/{len(top_factors)} positive key factors"
    
    full_message = f"{icon} **Key Factors Analysis:** {message}"
    
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
        icon = "📈"
        stability = "Stable"
    elif volatility < 0.5:
        alert_type = "warning"
        icon = "📊"
        stability = "Moderate"
    else:
        alert_type = "error"
        icon = "📉"
        stability = "Volatile"
    
    message = f"{icon} **Seasonal Analysis:** {stability} revenue pattern (Best: {best_month}, Worst: {worst_month}, Volatility: {volatility:.2f})"
    
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
        icon = "💰"
        status = "Healthy Cash Flow"
    elif negative_ratio < 0.25 or avg_balance > 1000:  # Less than 25% negative days or decent balance
        alert_type = "warning"
        icon = "⚠️"
        status = "Moderate Cash Flow Concerns"
    else:
        alert_type = "error"
        icon = "🚨"
        status = "Significant Cash Flow Issues"
    
    message = f"{icon} **{status}:** {negative_days}/{total_days} negative cash flow days, £{avg_balance:,.2f} average balance"
    
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
        icon = "✅"
        status = "Excellent Debt Management"
    elif dscr >= 1.2 and debt_ratio < 0.5:  # Acceptable DSCR and moderate debt ratio
        alert_type = "warning"
        icon = "⚠️"
        status = "Acceptable Debt Management"
    else:
        alert_type = "error"
        icon = "❌"
        status = "Debt Management Concerns"
    
    message = f"{icon} **{status}:** DSCR {dscr:.2f}, Debt Ratio {debt_ratio*100:.1f}%"
    
    if alert_type == "success":
        st.success(message)
    elif alert_type == "warning":
        st.warning(message)
    else:
        st.error(message)

def show_transaction_anomaly_alert(anomalies: list) -> None:
    """Display transaction anomaly detection alert."""
    
    if not anomalies:
        st.success("✅ **Transaction Analysis:** No significant anomalies detected")
    else:
        severity = len(anomalies)
        
        if severity <= 2:
            alert_type = "warning"
            icon = "⚠️"
            status = "Minor Anomalies"
        else:
            alert_type = "error"
            icon = "🚨"
            status = "Multiple Anomalies"
        
        message = f"{icon} **{status} Detected:** {severity} potential issues found"
        
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
        st.info("ℹ️ **No specific recommendations at this time**")
        return
    
    priority_config = {
        "high": {"color": "error", "icon": "🚨", "text": "Urgent Recommendations"},
        "medium": {"color": "warning", "icon": "💡", "text": "Recommendations"},
        "low": {"color": "info", "icon": "💭", "text": "Suggestions"}
    }
    
    config = priority_config.get(priority, priority_config["medium"])
    
    message = f"{config['icon']} **{config['text']}:** {len(recommendations)} action items identified"
    
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
        icon = "🟢"
    elif hours_since < 4:
        status = "Current"
        icon = "🟡"
    else:
        status = "Stale"
        icon = "🔴"
    
    st.info(f"{icon} **Cache Status:** {status} ({cache_size} items, last refresh: {hours_since:.1f}h ago)")

# Custom alert with custom styling
def show_custom_alert(
    message: str, 
    alert_type: str = "info", 
    icon: str = None, 
    expandable_content: Dict[str, Any] = None
) -> None:
    """Display a custom alert with flexible styling."""
    
    # Default icons for each type
    default_icons = {
        "success": "✅",
        "error": "❌", 
        "warning": "⚠️",
        "info": "ℹ️"
    }
    
    display_icon = icon or default_icons.get(alert_type, "ℹ️")
    full_message = f"{display_icon} {message}"
    
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
#!/usr/bin/env python3
"""
Scaler Diagnostic Script
Save this as: scaler_diagnostic.py
Run with: python scaler_diagnostic.py
"""

import joblib
import numpy as np
import os

def main():
    """Analyze the ML scaler file to understand training data distribution"""
    
    print("🔍 ML Scaler Diagnostic Tool")
    print("=" * 50)
    
    # Try to load the scaler
    scaler_path = "scaler.pkl"
    
    # Check if file exists
    if not os.path.exists(scaler_path):
        print(f"❌ Error: {scaler_path} not found in current directory")
        print(f"📁 Current directory: {os.getcwd()}")
        print(f"📋 Files in current directory:")
        for file in os.listdir("."):
            if file.endswith(('.pkl', '.joblib')):
                print(f"   • {file}")
        return
    
    try:
        # Load the scaler
        print(f"📥 Loading {scaler_path}...")
        scaler = joblib.load(scaler_path)
        print("✅ Scaler loaded successfully!")
        
        # Display basic info
        print(f"\n📊 Scaler Information:")
        print(f"   Scaler Type: {type(scaler).__name__}")
        print(f"   Number of Features: {len(scaler.mean_)}")
        
        # Feature names (based on your ML model)
        feature_names = [
            'Directors Score',
            'Total Revenue', 
            'Total Debt',
            'Debt-to-Income Ratio',
            'Operating Margin',
            'Debt Service Coverage Ratio',
            'Cash Flow Volatility',
            'Revenue Growth Rate',
            'Average Month-End Balance',
            'Average Negative Balance Days per Month',
            'Number of Bounced Payments',
            'Company Age (Months)',
            'Sector_Risk'
        ]
        
        print(f"\n🎯 Training Data Statistics:")
        print("=" * 70)
        print(f"{'Feature':<40} {'Mean':<15} {'Std Dev':<15}")
        print("-" * 70)
        
        for i, (name, mean, std) in enumerate(zip(feature_names, scaler.mean_, scaler.scale_)):
            if i < len(scaler.mean_):
                print(f"{name:<40} {mean:<15.3f} {std:<15.3f}")
        
        print("\n💡 What This Means:")
        print("   • Mean = Average value in training data")
        print("   • Std Dev = How much variation there was")
        print("   • Lower std dev = more consistent across businesses")
        print("   • Higher std dev = more variation across businesses")
        
        # Identify most/least variable features
        print(f"\n📈 Feature Variability Analysis:")
        
        # Calculate coefficient of variation (std/mean) where mean != 0
        cv_data = []
        for i, (name, mean, std) in enumerate(zip(feature_names, scaler.mean_, scaler.scale_)):
            if i < len(scaler.mean_) and abs(mean) > 0.001:  # Avoid division by zero
                cv = abs(std / mean)
                cv_data.append((name, cv, mean, std))
        
        # Sort by coefficient of variation
        cv_data.sort(key=lambda x: x[1])
        
        print("   Most Consistent Features (Low Variation):")
        for name, cv, mean, std in cv_data[:3]:
            print(f"   • {name}: CV = {cv:.2f}")
        
        print("   Most Variable Features (High Variation):")
        for name, cv, mean, std in cv_data[-3:]:
            print(f"   • {name}: CV = {cv:.2f}")
        
        # Business insights
        print(f"\n🏢 Business Insights from Training Data:")
        
        # Revenue insights
        revenue_mean = scaler.mean_[1] if len(scaler.mean_) > 1 else 0
        revenue_std = scaler.scale_[1] if len(scaler.scale_) > 1 else 0
        
        if revenue_mean > 0:
            print(f"   • Typical Business Revenue: £{revenue_mean:,.0f}")
            print(f"   • Revenue Range (±1 std): £{revenue_mean-revenue_std:,.0f} to £{revenue_mean+revenue_std:,.0f}")
        
        # DSCR insights
        dscr_mean = scaler.mean_[5] if len(scaler.mean_) > 5 else 0
        dscr_std = scaler.scale_[5] if len(scaler.scale_) > 5 else 0
        
        if dscr_mean > 0:
            print(f"   • Typical DSCR: {dscr_mean:.2f}")
            print(f"   • DSCR Range (±1 std): {dscr_mean-dscr_std:.2f} to {dscr_mean+dscr_std:.2f}")
        
        # Directors Score insights
        dir_mean = scaler.mean_[0] if len(scaler.mean_) > 0 else 0
        dir_std = scaler.scale_[0] if len(scaler.scale_) > 0 else 0
        
        if dir_mean > 0:
            print(f"   • Typical Directors Score: {dir_mean:.0f}")
            print(f"   • Directors Score Range (±1 std): {dir_mean-dir_std:.0f} to {dir_mean+dir_std:.0f}")
        
        print(f"\n✅ Analysis Complete!")
        print(f"💡 This data can help calibrate your scorecard thresholds")
        
    except Exception as e:
        print(f"❌ Error loading scaler: {e}")
        print(f"🔧 Make sure the file is a valid joblib/pickle file")

if __name__ == "__main__":
    main()
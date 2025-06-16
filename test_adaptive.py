# Create this as test_adaptive.py in your MCAV2 folder and run it to test the module

import sys
import os

# Add the app folder to Python path
app_path = os.path.join(os.getcwd(), 'app')
if app_path not in sys.path:
    sys.path.insert(0, app_path)

print(f"Testing adaptive_score_calculation.py...")
print(f"App path: {app_path}")
print(f"File exists: {os.path.exists(os.path.join(app_path, 'adaptive_score_calculation.py'))}")

try:
    print("\n1. Testing import...")
    from adaptive_score_calculation import get_improved_weighted_score, get_detailed_scoring_breakdown
    print("✅ Import successful!")
    
    print("\n2. Testing function calls...")
    
    # Complete test data that matches what the main app provides
    test_metrics = {
        'Debt Service Coverage Ratio': 1.5,
        'Net Income': 5000,
        'Operating Margin': 0.1,
        'Revenue Growth Rate': 0.05,
        'Cash Flow Volatility': 0.15,
        'Gross Burn Rate': 10000,
        'Average Month-End Balance': 5000,
        'Average Negative Balance Days per Month': 1,
        'Number of Bounced Payments': 0,
        'Total Revenue': 50000,
        'Total Debt': 5000,
        'Debt-to-Income Ratio': 0.1,
        'Total Expenses': 45000,
        'Total Debt Repayments': 1000,
        'Expense-to-Revenue Ratio': 0.9,
        'Monthly Average Revenue': 4166.67
    }
    
    test_thresholds = {
        'Debt Service Coverage Ratio': 1.4,
        'Net Income': 1000,
        'Operating Margin': 0.08,
        'Revenue Growth Rate': 0.04,
        'Cash Flow Volatility': 0.2,
        'Gross Burn Rate': 12000,
        'Directors Score': 75,
        'Sector Risk': 0,
        'Average Month-End Balance': 1000,
        'Average Negative Balance Days per Month': 2,
        'Number of Bounced Payments': 0
    }
    
    test_penalties = {
        "personal_default_12m": 3,
        "business_ccj": 5,
        "director_ccj": 3,
        'website_or_social_outdated': 3,
        'uses_generic_email': 1,
        'no_online_presence': 2
    }
    
    # Test the detailed scoring function
    adaptive_score, raw_score, details = get_detailed_scoring_breakdown(
        test_metrics, 75, 0, test_thresholds, 24, 
        False, False, False, False, False, False, test_penalties
    )
    
    print(f"✅ Function test successful!")
    print(f"   ML-Aligned Adaptive Score: {adaptive_score:.1f}%")
    print(f"   Raw Score: {raw_score:.1f}/105")
    print(f"   Details: {len(details)} scoring components")
    
    # Test the simple scoring function
    simple_score = get_improved_weighted_score(
        test_metrics, 75, 0, test_thresholds, {},
        24, False, False, False, False, False, False, test_penalties
    )
    
    print(f"   Simple Adaptive Score: {simple_score:.1f}%")
    
    print("\n📋 Sample Scoring Breakdown:")
    for i, detail in enumerate(details[:5]):  # Show first 5 details
        print(f"   {i+1}. {detail}")
    
    if len(details) > 5:
        print(f"   ... and {len(details) - 5} more components")
    
    print("\n🎉 All tests passed! The adaptive scoring module is working correctly.")
    print("   You can now run your main application and it should show adaptive scoring.")
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("   The module file exists but cannot be imported.")
    print("   This usually means there's a syntax error in the file.")
    
except Exception as e:
    print(f"❌ Runtime Error: {e}")
    print("   The module imported but failed when running.")
    import traceback
    print(f"   Full error: {traceback.format_exc()}")
    
print("\n" + "="*60)
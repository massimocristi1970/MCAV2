# app/utils/weight_calibration.py
"""
Weight Calibration Utility

Derives optimal scoring weights from trained ML model coefficients.
This ensures rule-based scoring aligns with ML feature importance.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import joblib
from pathlib import Path


class WeightCalibrator:
    """
    Calibrates scoring weights based on ML model feature importance.
    
    Uses logistic regression coefficients or feature importance scores
    to derive empirically-grounded weights for rule-based scoring.
    """
    
    # Default feature mapping between ML and rule-based systems
    FEATURE_MAPPING = {
        # ML Feature Name -> Rule-based metric name
        'Directors Score': 'Directors Score',
        'Total Revenue': 'Total Revenue',
        'Total Debt': 'Total Debt',
        'Debt-to-Income Ratio': 'Debt-to-Income Ratio',
        'Operating Margin': 'Operating Margin',
        'Debt Service Coverage Ratio': 'Debt Service Coverage Ratio',
        'Cash Flow Volatility': 'Cash Flow Volatility',
        'Revenue Growth Rate': 'Revenue Growth Rate',
        'Average Month-End Balance': 'Average Month-End Balance',
        'Average Negative Balance Days per Month': 'Average Negative Balance Days per Month',
        'Number of Bounced Payments': 'Number of Bounced Payments',
        'Company Age (Months)': 'Company Age (Months)',
        'Sector_Risk': 'Sector Risk',
    }
    
    def __init__(self, model_path: str = 'model.pkl', scaler_path: str = 'scaler.pkl'):
        """
        Initialize calibrator with model paths.
        
        Args:
            model_path: Path to the trained model file
            scaler_path: Path to the feature scaler file
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.feature_names = None
        
    def load_model(self) -> bool:
        """
        Load the trained model and scaler.
        
        Returns:
            True if loading successful, False otherwise
        """
        try:
            model_file = Path(self.model_path)
            scaler_file = Path(self.scaler_path)
            
            if not model_file.exists():
                print(f"Model file not found: {model_file}")
                return False
            
            self.model = joblib.load(model_file)
            
            if scaler_file.exists():
                self.scaler = joblib.load(scaler_file)
                
                # Get feature names from scaler if available
                if hasattr(self.scaler, 'feature_names_in_'):
                    self.feature_names = list(self.scaler.feature_names_in_)
            
            # Fallback feature names
            if self.feature_names is None:
                self.feature_names = list(self.FEATURE_MAPPING.keys())
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def extract_coefficients(self) -> Optional[Dict[str, float]]:
        """
        Extract feature coefficients from trained model.
        
        Returns:
            Dictionary mapping feature names to coefficients, or None if unavailable
        """
        if self.model is None:
            if not self.load_model():
                return None
        
        coefficients = {}
        
        # Try to get coefficients from different model types
        if hasattr(self.model, 'coef_'):
            # Logistic Regression, Linear SVM, etc.
            coefs = self.model.coef_[0] if len(self.model.coef_.shape) > 1 else self.model.coef_
            for i, name in enumerate(self.feature_names):
                if i < len(coefs):
                    coefficients[name] = float(coefs[i])
                    
        elif hasattr(self.model, 'feature_importances_'):
            # Random Forest, Gradient Boosting, etc.
            for i, name in enumerate(self.feature_names):
                if i < len(self.model.feature_importances_):
                    coefficients[name] = float(self.model.feature_importances_[i])
                    
        elif hasattr(self.model, 'estimators_'):
            # Ensemble methods - average importance across estimators
            importances = np.zeros(len(self.feature_names))
            for estimator in self.model.estimators_:
                if hasattr(estimator, 'feature_importances_'):
                    importances += estimator.feature_importances_
            importances /= len(self.model.estimators_)
            
            for i, name in enumerate(self.feature_names):
                if i < len(importances):
                    coefficients[name] = float(importances[i])
        
        return coefficients if coefficients else None
    
    def calibrate_weights(
        self,
        target_total: int = 100,
        min_weight: int = 2,
        max_weight: int = 25
    ) -> Dict[str, int]:
        """
        Calibrate rule-based weights from ML coefficients.
        
        Args:
            target_total: Target sum of all weights (default 100)
            min_weight: Minimum weight for any feature
            max_weight: Maximum weight for any feature
            
        Returns:
            Dictionary mapping feature names to calibrated integer weights
        """
        coefficients = self.extract_coefficients()
        
        if coefficients is None:
            print("Could not extract coefficients. Using default weights.")
            return self._get_default_weights(target_total)
        
        # Use absolute values of coefficients for importance
        abs_coefficients = {k: abs(v) for k, v in coefficients.items()}
        total_importance = sum(abs_coefficients.values())
        
        if total_importance == 0:
            return self._get_default_weights(target_total)
        
        # Calculate raw proportional weights
        raw_weights = {
            k: (v / total_importance) * target_total 
            for k, v in abs_coefficients.items()
        }
        
        # Apply min/max constraints and round to integers
        calibrated_weights = {}
        for feature, raw_weight in raw_weights.items():
            weight = int(round(max(min_weight, min(max_weight, raw_weight))))
            # Map to rule-based metric name if different
            metric_name = self.FEATURE_MAPPING.get(feature, feature)
            calibrated_weights[metric_name] = weight
        
        # Adjust to hit target total
        current_total = sum(calibrated_weights.values())
        adjustment = target_total - current_total
        
        if adjustment != 0:
            # Distribute adjustment across features proportionally
            sorted_features = sorted(
                calibrated_weights.keys(),
                key=lambda k: raw_weights.get(self._reverse_mapping(k), 0),
                reverse=(adjustment > 0)
            )
            
            for feature in sorted_features:
                if adjustment == 0:
                    break
                    
                current = calibrated_weights[feature]
                if adjustment > 0 and current < max_weight:
                    calibrated_weights[feature] = current + 1
                    adjustment -= 1
                elif adjustment < 0 and current > min_weight:
                    calibrated_weights[feature] = current - 1
                    adjustment += 1
        
        return calibrated_weights
    
    def _reverse_mapping(self, metric_name: str) -> str:
        """Get ML feature name from rule-based metric name."""
        for ml_name, rule_name in self.FEATURE_MAPPING.items():
            if rule_name == metric_name:
                return ml_name
        return metric_name
    
    def _get_default_weights(self, target_total: int) -> Dict[str, int]:
        """Return default equal weights if calibration fails."""
        num_features = len(self.FEATURE_MAPPING)
        base_weight = target_total // num_features
        remainder = target_total % num_features
        
        weights = {}
        for i, (_, metric_name) in enumerate(self.FEATURE_MAPPING.items()):
            weights[metric_name] = base_weight + (1 if i < remainder else 0)
        
        return weights
    
    def compare_weights(
        self,
        current_weights: Dict[str, int],
        calibrated_weights: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Compare current weights with calibrated weights.
        
        Args:
            current_weights: Currently used weights
            calibrated_weights: Weights derived from ML model
            
        Returns:
            Comparison analysis
        """
        comparison = {
            'features': [],
            'total_divergence': 0,
            'recommendations': []
        }
        
        all_features = set(current_weights.keys()) | set(calibrated_weights.keys())
        
        for feature in sorted(all_features):
            current = current_weights.get(feature, 0)
            calibrated = calibrated_weights.get(feature, 0)
            difference = calibrated - current
            
            comparison['features'].append({
                'feature': feature,
                'current_weight': current,
                'calibrated_weight': calibrated,
                'difference': difference,
                'percentage_change': (difference / current * 100) if current > 0 else 0
            })
            
            comparison['total_divergence'] += abs(difference)
            
            # Generate recommendations for significant differences
            if abs(difference) >= 3:
                if difference > 0:
                    comparison['recommendations'].append(
                        f"Consider increasing '{feature}' weight from {current} to {calibrated}"
                    )
                else:
                    comparison['recommendations'].append(
                        f"Consider decreasing '{feature}' weight from {current} to {calibrated}"
                    )
        
        comparison['alignment_score'] = max(0, 100 - comparison['total_divergence'])
        
        return comparison
    
    def get_feature_importance_report(self) -> Dict[str, Any]:
        """
        Generate a detailed feature importance report.
        
        Returns:
            Report with feature rankings and insights
        """
        coefficients = self.extract_coefficients()
        
        if coefficients is None:
            return {'error': 'Could not extract coefficients'}
        
        # Sort by absolute importance
        sorted_features = sorted(
            coefficients.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        total_importance = sum(abs(v) for v in coefficients.values())
        
        report = {
            'model_type': type(self.model).__name__ if self.model else 'Unknown',
            'feature_count': len(coefficients),
            'rankings': [],
            'top_3_features': [],
            'bottom_3_features': []
        }
        
        for rank, (feature, coef) in enumerate(sorted_features, 1):
            importance_pct = (abs(coef) / total_importance * 100) if total_importance > 0 else 0
            direction = 'positive' if coef > 0 else 'negative'
            
            entry = {
                'rank': rank,
                'feature': feature,
                'coefficient': round(coef, 4),
                'absolute_importance': round(abs(coef), 4),
                'importance_percentage': round(importance_pct, 1),
                'direction': direction
            }
            
            report['rankings'].append(entry)
            
            if rank <= 3:
                report['top_3_features'].append(entry)
            if rank > len(sorted_features) - 3:
                report['bottom_3_features'].append(entry)
        
        return report


def calibrate_weights_from_model(
    model_path: str = 'model.pkl',
    scaler_path: str = 'scaler.pkl',
    target_total: int = 100
) -> Dict[str, int]:
    """
    Convenience function to calibrate weights from a model file.
    
    Args:
        model_path: Path to the trained model
        scaler_path: Path to the feature scaler
        target_total: Target sum of all weights
        
    Returns:
        Calibrated weights dictionary
    """
    calibrator = WeightCalibrator(model_path, scaler_path)
    return calibrator.calibrate_weights(target_total)


def get_weight_comparison_report(
    current_weights: Dict[str, int],
    model_path: str = 'model.pkl',
    scaler_path: str = 'scaler.pkl'
) -> Dict[str, Any]:
    """
    Generate a comparison report between current and ML-calibrated weights.
    
    Args:
        current_weights: Currently used weights
        model_path: Path to the trained model
        scaler_path: Path to the feature scaler
        
    Returns:
        Comparison report
    """
    calibrator = WeightCalibrator(model_path, scaler_path)
    calibrated = calibrator.calibrate_weights()
    return calibrator.compare_weights(current_weights, calibrated)

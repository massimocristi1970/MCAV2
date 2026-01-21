# app/models/ml_predictor.py
"""Enhanced ML prediction service with explainability and confidence intervals."""

import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import warnings

from ..core.exceptions import ModelPredictionError
from ..core.logger import get_logger, log_performance
from ..core.cache import CacheManager
from ..config.settings import settings

logger = get_logger("ml_predictor")

class MLPredictor:
    """Enhanced ML prediction service with model explainability."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self._load_model_artifacts()
    
    @CacheManager.cache_resource()
    def _load_model_artifacts(self) -> None:
        """Load model and scaler from disk."""
        try:
            model_path = Path(settings.BASE_DIR) / settings.MODEL_PATH
            scaler_path = Path(settings.BASE_DIR) / settings.SCALER_PATH
            
            if not model_path.exists():
                raise ModelPredictionError(f"Model file not found: {model_path}")
            
            if not scaler_path.exists():
                raise ModelPredictionError(f"Scaler file not found: {scaler_path}")
            
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            
            # Extract feature names from scaler if available
            if hasattr(self.scaler, 'feature_names_in_'):
                self.feature_names = self.scaler.feature_names_in_
            else:
                # Fallback feature names based on your original code
                self.feature_names = [
                    'Directors Score', 'Total Revenue', 'Total Debt',
                    'Debt-to-Income Ratio', 'Operating Margin',
                    'Debt Service Coverage Ratio', 'Cash Flow Volatility',
                    'Revenue Growth Rate', 'Average Month-End Balance',
                    'Average Negative Balance Days per Month',
                    'Number of Bounced Payments', 'Company Age (Months)',
                    'Sector_Risk'
                ]
            
            logger.info("Model and scaler loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model artifacts: {str(e)}")
            raise ModelPredictionError(f"Failed to load model: {str(e)}")
    
    def _prepare_features(
        self, 
        metrics: Dict[str, Any], 
        directors_score: int, 
        sector_risk: int, 
        company_age_months: int
    ) -> pd.DataFrame:
        """Prepare features for model prediction."""
        
        features = {
            'Directors Score': directors_score,
            'Total Revenue': metrics.get("Total Revenue", 0),
            'Total Debt': metrics.get("Total Debt", 0),
            'Debt-to-Income Ratio': metrics.get("Debt-to-Income Ratio", 0),
            'Operating Margin': metrics.get("Operating Margin", 0),
            'Debt Service Coverage Ratio': metrics.get("Debt Service Coverage Ratio", 0),
            'Cash Flow Volatility': metrics.get("Cash Flow Volatility", 0),
            'Revenue Growth Rate': metrics.get("Revenue Growth Rate", 0),
            'Average Month-End Balance': metrics.get("Average Month-End Balance", 0),
            'Average Negative Balance Days per Month': metrics.get("Average Negative Balance Days per Month", 0),
            'Number of Bounced Payments': metrics.get("Number of Bounced Payments", 0),
            'Company Age (Months)': company_age_months,
            'Sector_Risk': sector_risk
        }
        
        # Create DataFrame
        features_df = pd.DataFrame([features])
        
        # Clean the data
        features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        features_df.fillna(0, inplace=True)
        
        # Ensure all expected features are present
        for feature_name in self.feature_names:
            if feature_name not in features_df.columns:
                features_df[feature_name] = 0
        
        # Reorder columns to match training data
        features_df = features_df[self.feature_names]
        
        return features_df
    
    @log_performance(logger)
    def predict_repayment_probability(
        self, 
        metrics: Dict[str, Any], 
        directors_score: int, 
        sector_risk: int, 
        company_age_months: int,
        include_confidence: bool = True
    ) -> Dict[str, Any]:
        """
        Predict repayment probability with confidence intervals and feature importance.
        
        Returns:
            Dictionary containing prediction, confidence intervals, and explanations
        """
        try:
            # Prepare features
            features_df = self._prepare_features(
                metrics, directors_score, sector_risk, company_age_months
            )
            
            # Scale features
            features_scaled = self.scaler.transform(features_df)
            
            # Make prediction
            probability = self.model.predict_proba(features_scaled)[:, 1][0]
            prediction_score = probability * 100
            
            result = {
                'probability': round(prediction_score, 2),
                'risk_category': self._categorize_risk(prediction_score),
                'model_confidence': self._calculate_confidence(features_scaled),
            }
            
            if include_confidence:
                # Calculate confidence intervals using bootstrap if available
                confidence_interval = self._calculate_confidence_interval(features_scaled)
                result['confidence_interval'] = confidence_interval
                
                # Feature importance for this prediction
                feature_importance = self._get_feature_importance(features_df.iloc[0])
                result['feature_importance'] = feature_importance
                
                # Model explanation
                result['explanation'] = self._generate_explanation(
                    features_df.iloc[0], prediction_score, feature_importance
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise ModelPredictionError(f"Prediction failed: {str(e)}")
    
    def _categorize_risk(self, probability: float) -> str:
        """Categorize risk based on probability score."""
        if probability >= 80:
            return "Very Low Risk"
        elif probability >= 65:
            return "Low Risk"
        elif probability >= 50:
            return "Moderate Risk"
        elif probability >= 35:
            return "High Risk"
        else:
            return "Very High Risk"
    
    def _calculate_confidence(self, features_scaled: np.ndarray) -> float:
        """
        Calculate model confidence based on prediction certainty.
        
        Uses multiple factors:
        - Probability margin (distance from decision boundary)
        - Feature value extremity (unusual values = lower confidence)
        """
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Base confidence from probability margin
        sorted_probs = np.sort(probabilities)
        margin_confidence = (sorted_probs[-1] - sorted_probs[-2]) * 100
        
        # Adjust for extreme predictions (very high/low probs are less certain)
        base_prob = probabilities[1] if len(probabilities) > 1 else probabilities[0]
        extremity_penalty = abs(base_prob - 0.5) * 20  # Max 10% penalty at extremes
        
        # Final confidence (margin gives certainty, extremity reduces it slightly)
        confidence = margin_confidence - extremity_penalty
        
        return round(max(10, min(95, confidence)), 2)
    
    def _calculate_confidence_interval(
        self, 
        features_scaled: np.ndarray, 
        n_bootstrap: int = 100
    ) -> Dict[str, float]:
        """
        Calculate confidence intervals with improved uncertainty estimation.
        
        Uses adaptive uncertainty based on:
        - Probability extremity (predictions near 0.5 have higher uncertainty)
        - Model type (ensemble models have better-calibrated uncertainty)
        - Feature scaling factors
        """
        try:
            base_prob = self.model.predict_proba(features_scaled)[:, 1][0]
            
            # For ensemble models, use built-in variance if available
            if hasattr(self.model, 'estimators_'):
                # Ensemble method - calculate actual variance across trees
                predictions = []
                for estimator in self.model.estimators_[:min(50, len(self.model.estimators_))]:
                    try:
                        pred = estimator.predict_proba(features_scaled)[:, 1][0]
                        predictions.append(pred)
                    except:
                        continue
                
                if len(predictions) > 1:
                    std_dev = np.std(predictions)
                    lower_bound = max(0, base_prob - 1.96 * std_dev) * 100
                    upper_bound = min(1, base_prob + 1.96 * std_dev) * 100
                    
                    return {
                        'lower': round(lower_bound, 2),
                        'upper': round(upper_bound, 2),
                        'width': round(upper_bound - lower_bound, 2),
                        'method': 'ensemble_variance',
                        'n_estimators_used': len(predictions)
                    }
            
            # For non-ensemble models, use adaptive uncertainty
            # Uncertainty is higher near decision boundary (0.5) and lower at extremes
            # This reflects actual prediction reliability patterns
            
            # Base uncertainty varies with distance from boundary
            distance_from_boundary = abs(base_prob - 0.5)
            
            # Sigmoid-shaped uncertainty: highest at 0.5, decreases toward 0 and 1
            # But never goes to zero (there's always some uncertainty)
            min_std = 0.02  # Minimum 2% std dev
            max_std = 0.12  # Maximum 12% std dev at boundary
            
            # Uncertainty decreases as we move away from boundary
            std_dev = max_std - (distance_from_boundary * (max_std - min_std) / 0.5)
            std_dev = max(min_std, min(max_std, std_dev))
            
            # Additional uncertainty for feature extremity
            feature_extremity = self._calculate_feature_extremity(features_scaled)
            std_dev *= (1 + feature_extremity * 0.3)  # Up to 30% increase for extreme features
            
            lower_bound = max(0, base_prob - 1.96 * std_dev) * 100
            upper_bound = min(1, base_prob + 1.96 * std_dev) * 100
            
            return {
                'lower': round(lower_bound, 2),
                'upper': round(upper_bound, 2),
                'width': round(upper_bound - lower_bound, 2),
                'method': 'adaptive_uncertainty',
                'estimated_std': round(std_dev * 100, 2)
            }
            
        except Exception as e:
            logger.warning(f"Could not calculate confidence interval: {str(e)}")
            return {'lower': 0, 'upper': 100, 'width': 100, 'method': 'fallback'}
    
    def _calculate_feature_extremity(self, features_scaled: np.ndarray) -> float:
        """
        Calculate how extreme the feature values are.
        
        Features more than 2 std devs from mean indicate unusual cases
        where the model may be less reliable.
        
        Returns:
            Extremity score from 0 (typical) to 1 (very unusual)
        """
        try:
            # Scaled features should be roughly N(0,1) if StandardScaler was used
            # Count features that are more than 2 std devs from 0
            extreme_count = np.sum(np.abs(features_scaled) > 2)
            total_features = features_scaled.size
            
            # Proportion of extreme features
            extremity = extreme_count / total_features if total_features > 0 else 0
            
            return min(1.0, extremity)
            
        except:
            return 0.0
    
    def _get_feature_importance(self, features: pd.Series) -> Dict[str, float]:
        """Get feature importance for this specific prediction."""
        try:
            # For logistic regression, use coefficients as importance
            if hasattr(self.model, 'coef_'):
                coefficients = self.model.coef_[0]
                
                # Calculate importance as absolute coefficient * feature value
                importance_scores = {}
                for i, feature_name in enumerate(self.feature_names):
                    importance = abs(coefficients[i] * features[feature_name])
                    importance_scores[feature_name] = importance
                
                # Normalize to percentages
                total_importance = sum(importance_scores.values())
                if total_importance > 0:
                    importance_scores = {
                        k: round((v / total_importance) * 100, 2)
                        for k, v in importance_scores.items()
                    }
                
                # Sort by importance
                return dict(
                    sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
                )
            
        except Exception as e:
            logger.warning(f"Could not calculate feature importance: {str(e)}")
        
        return {}
    
    def _generate_explanation(
        self, 
        features: pd.Series, 
        probability: float, 
        feature_importance: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generate human-readable explanation of the prediction."""
        
        explanation = {
            'summary': f"Based on the analysis, this business has a {probability:.1f}% probability of successful loan repayment.",
            'key_factors': [],
            'risk_factors': [],
            'recommendations': []
        }
        
        # Identify top contributing factors
        top_factors = list(feature_importance.items())[:3]
        
        for factor_name, importance in top_factors:
            factor_value = features[factor_name]
            
            if factor_name == 'Directors Score':
                if factor_value >= 75:
                    explanation['key_factors'].append(
                        f"Strong director credit score ({factor_value}/100) - {importance:.1f}% impact"
                    )
                else:
                    explanation['risk_factors'].append(
                        f"Low director credit score ({factor_value}/100) - {importance:.1f}% impact"
                    )
            
            elif factor_name == 'Debt Service Coverage Ratio':
                if factor_value >= 1.5:
                    explanation['key_factors'].append(
                        f"Healthy debt service coverage ratio ({factor_value:.2f}) - {importance:.1f}% impact"
                    )
                else:
                    explanation['risk_factors'].append(
                        f"Low debt service coverage ratio ({factor_value:.2f}) - {importance:.1f}% impact"
                    )
            
            elif factor_name == 'Operating Margin':
                if factor_value >= 0.1:
                    explanation['key_factors'].append(
                        f"Strong operating margin ({factor_value*100:.1f}%) - {importance:.1f}% impact"
                    )
                else:
                    explanation['risk_factors'].append(
                        f"Low operating margin ({factor_value*100:.1f}%) - {importance:.1f}% impact"
                    )
        
        # Generate recommendations
        if probability < 50:
            explanation['recommendations'].extend([
                "Consider requiring additional collateral or guarantees",
                "Implement enhanced monitoring and reporting requirements",
                "Consider a smaller loan amount or shorter term"
            ])
        elif probability < 70:
            explanation['recommendations'].extend([
                "Standard loan terms with regular monitoring",
                "Consider quarterly financial reviews"
            ])
        else:
            explanation['recommendations'].extend([
                "Business shows strong financial health",
                "Standard loan terms appropriate"
            ])
        
        return explanation
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_type': type(self.model).__name__,
            'feature_count': len(self.feature_names),
            'features': self.feature_names,
            'scaler_type': type(self.scaler).__name__
        }

# Global predictor instance
ml_predictor = MLPredictor()
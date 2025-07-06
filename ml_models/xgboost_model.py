"""
XGBoost Model for Trading Signal Prediction
"""
import pandas as pd
import numpy as np
from typing import Dict, Any
import xgboost as xgb
from .base_model import BaseMLModel
from loguru import logger


class XGBoostTradingModel(BaseMLModel):
    """XGBoost model for trading signal prediction"""
    
    def __init__(self, model_dir):
        """Initialize XGBoost model"""
        super().__init__("XGBoost", model_dir)
        
        # Default XGBoost parameters optimized for trading
        self.default_params = {
            'objective': 'multi:softprob',
            'num_class': 3,  # -1: sell, 0: hold, 1: buy
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist',  # Faster training
            'eval_metric': 'mlogloss'
        }
    
    def _create_model(self, **kwargs) -> xgb.XGBClassifier:
        """Create XGBoost classifier"""
        # Merge default params with custom params
        params = {**self.default_params, **kwargs}
        
        logger.info(f"Creating XGBoost model with params: {params}")
        return xgb.XGBClassifier(**params)
    
    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit XGBoost model"""
        logger.info(f"Training XGBoost on {X.shape[0]} samples with {X.shape[1]} features")
        
        # Fit the model
        self.model.fit(
            X, y,
            eval_set=[(X, y)],
            verbose=False
        )
        
        logger.info("XGBoost training completed")
    
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get XGBoost feature importance"""
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return {}
        
        importance_dict = {}
        for i, importance in enumerate(self.model.feature_importances_):
            if i < len(self.feature_names):
                importance_dict[self.feature_names[i]] = float(importance)
        
        return importance_dict
    
    def get_detailed_feature_importance(self) -> Dict[str, Any]:
        """Get detailed feature importance metrics"""
        if not self.is_trained:
            return {}
        
        # Get different importance types
        importance_types = ['weight', 'gain', 'cover']
        detailed_importance = {}
        
        for imp_type in importance_types:
            try:
                importance_dict = self.model.get_booster().get_score(importance_type=imp_type)
                # Map feature names
                mapped_importance = {}
                for feat_name, importance in importance_dict.items():
                    # XGBoost uses f0, f1, f2... format
                    feat_idx = int(feat_name[1:]) if feat_name.startswith('f') else 0
                    if feat_idx < len(self.feature_names):
                        mapped_importance[self.feature_names[feat_idx]] = importance
                
                detailed_importance[imp_type] = mapped_importance
            except Exception as e:
                logger.warning(f"Could not get {imp_type} importance: {e}")
        
        return detailed_importance
    
    def optimize_hyperparameters(self, features: pd.DataFrame, target: pd.Series,
                                cv_folds: int = 5) -> Dict[str, Any]:
        """
        Optimize hyperparameters using cross-validation
        
        Args:
            features: Training features
            target: Training target
            cv_folds: Number of CV folds
            
        Returns:
            Best parameters and CV results
        """
        from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
        from scipy.stats import randint, uniform
        
        logger.info("Starting hyperparameter optimization...")
        
        # Prepare data
        X, y = self.prepare_features(features, target, fit_scalers=True)
        
        # Define parameter search space
        param_distributions = {
            'max_depth': randint(3, 10),
            'learning_rate': uniform(0.01, 0.2),
            'n_estimators': randint(100, 500),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(0, 1),
        }
        
        # Create base model
        base_model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        )
        
        # Use TimeSeriesSplit for time series data
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Randomized search
        search = RandomizedSearchCV(
            base_model,
            param_distributions,
            n_iter=50,  # Number of parameter combinations to try
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        # Fit search
        search.fit(X, y)
        
        # Update model with best parameters
        self.default_params.update(search.best_params_)
        
        results = {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_
        }
        
        logger.info(f"Hyperparameter optimization completed. Best score: {search.best_score_:.4f}")
        
        return results
    
    def get_prediction_confidence(self, features: pd.DataFrame) -> np.ndarray:
        """
        Get prediction confidence scores
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Confidence scores (max probability for each prediction)
        """
        probabilities = self.predict_proba(features)
        return np.max(probabilities, axis=1)
    
    def explain_prediction(self, features: pd.DataFrame, sample_idx: int = 0) -> Dict[str, Any]:
        """
        Explain a single prediction using SHAP values (if available)
        
        Args:
            features: Feature DataFrame
            sample_idx: Index of sample to explain
            
        Returns:
            Explanation dictionary
        """
        try:
            import shap
            
            # Prepare features
            X, _ = self.prepare_features(features, fit_scalers=False)
            
            # Create explainer
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X[sample_idx:sample_idx+1])
            
            # Get base prediction
            prediction = self.predict(features.iloc[sample_idx:sample_idx+1])[0]
            probabilities = self.predict_proba(features.iloc[sample_idx:sample_idx+1])[0]
            
            explanation = {
                'prediction': prediction,
                'probabilities': probabilities.tolist(),
                'feature_contributions': {}
            }
            
            # Map SHAP values to feature names
            for i, feature_name in enumerate(self.feature_names):
                if i < len(shap_values[0][0]):
                    explanation['feature_contributions'][feature_name] = float(shap_values[0][0][i])
            
            return explanation
            
        except ImportError:
            logger.warning("SHAP not available for prediction explanation")
            return {
                'prediction': self.predict(features.iloc[sample_idx:sample_idx+1])[0],
                'probabilities': self.predict_proba(features.iloc[sample_idx:sample_idx+1])[0].tolist(),
                'note': 'Install SHAP for detailed explanations'
            }
        except Exception as e:
            logger.error(f"Error explaining prediction: {e}")
            return {}
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        summary = self.get_model_info()
        
        if self.is_trained:
            summary.update({
                'model_type': 'XGBoost Classifier',
                'n_estimators': self.model.n_estimators,
                'max_depth': self.model.max_depth,
                'learning_rate': self.model.learning_rate,
                'detailed_feature_importance': self.get_detailed_feature_importance()
            })
        
        return summary
"""
LightGBM Model for Trading Signal Prediction
"""
import pandas as pd
import numpy as np
from typing import Dict, Any
import lightgbm as lgb
from .base_model import BaseMLModel
from loguru import logger


class LightGBMTradingModel(BaseMLModel):
    """LightGBM model for trading signal prediction"""
    
    def __init__(self, model_dir):
        """Initialize LightGBM model"""
        super().__init__("LightGBM", model_dir)
        
        # Default LightGBM parameters optimized for trading
        self.default_params = {
            'objective': 'multiclass',
            'num_class': 3,  # -1: sell, 0: hold, 1: buy
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
            'force_col_wise': True  # Better for small datasets
        }
    
    def _create_model(self, **kwargs) -> lgb.LGBMClassifier:
        """Create LightGBM classifier"""
        # Merge default params with custom params
        params = {**self.default_params, **kwargs}
        
        logger.info(f"Creating LightGBM model with params: {params}")
        return lgb.LGBMClassifier(**params)
    
    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit LightGBM model"""
        logger.info(f"Training LightGBM on {X.shape[0]} samples with {X.shape[1]} features")
        
        # Fit the model
        self.model.fit(
            X, y,
            eval_set=[(X, y)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        logger.info("LightGBM training completed")
    
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get LightGBM feature importance"""
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
        importance_types = ['split', 'gain']
        detailed_importance = {}
        
        for imp_type in importance_types:
            try:
                if imp_type == 'split':
                    importances = self.model.booster_.feature_importance(importance_type='split')
                else:
                    importances = self.model.booster_.feature_importance(importance_type='gain')
                
                # Map to feature names
                mapped_importance = {}
                for i, importance in enumerate(importances):
                    if i < len(self.feature_names):
                        mapped_importance[self.feature_names[i]] = float(importance)
                
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
        
        logger.info("Starting LightGBM hyperparameter optimization...")
        
        # Prepare data
        X, y = self.prepare_features(features, target, fit_scalers=True)
        
        # Define parameter search space
        param_distributions = {
            'num_leaves': randint(20, 100),
            'learning_rate': uniform(0.01, 0.2),
            'n_estimators': randint(100, 500),
            'feature_fraction': uniform(0.6, 0.4),
            'bagging_fraction': uniform(0.6, 0.4),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(0, 1),
            'min_child_samples': randint(10, 50)
        }
        
        # Create base model
        base_model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=3,
            metric='multi_logloss',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        # Use TimeSeriesSplit for time series data
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Randomized search
        search = RandomizedSearchCV(
            base_model,
            param_distributions,
            n_iter=50,
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
        
        logger.info(f"LightGBM hyperparameter optimization completed. Best score: {search.best_score_:.4f}")
        
        return results
    
    def get_prediction_confidence(self, features: pd.DataFrame) -> np.ndarray:
        """Get prediction confidence scores"""
        probabilities = self.predict_proba(features)
        return np.max(probabilities, axis=1)
    
    def explain_prediction(self, features: pd.DataFrame, sample_idx: int = 0) -> Dict[str, Any]:
        """Explain a single prediction using SHAP values"""
        try:
            import shap
            
            # Prepare features
            X, _ = self.prepare_features(features, fit_scalers=False)
            
            # Create explainer
            explainer = shap.TreeExplainer(self.model.booster_)
            shap_values = explainer.shap_values(X[sample_idx:sample_idx+1])
            
            # Get base prediction
            prediction = self.predict(features.iloc[sample_idx:sample_idx+1])[0]
            probabilities = self.predict_proba(features.iloc[sample_idx:sample_idx+1])[0]
            
            explanation = {
                'prediction': prediction,
                'probabilities': probabilities.tolist(),
                'feature_contributions': {}
            }
            
            # For multiclass, SHAP returns list of arrays
            if isinstance(shap_values, list) and len(shap_values) > 0:
                # Use the SHAP values for the predicted class
                predicted_class = np.argmax(probabilities)
                class_shap_values = shap_values[predicted_class][0]
                
                for i, feature_name in enumerate(self.feature_names):
                    if i < len(class_shap_values):
                        explanation['feature_contributions'][feature_name] = float(class_shap_values[i])
            
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
    
    def plot_feature_importance(self, top_n: int = 20) -> None:
        """Plot feature importance"""
        try:
            import matplotlib.pyplot as plt
            
            if not self.is_trained:
                logger.warning("Model must be trained to plot feature importance")
                return
            
            # Get feature importance
            importance = self.get_feature_importance()
            if not importance:
                logger.warning("No feature importance available")
                return
            
            # Sort and select top features
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
            features, importances = zip(*sorted_features)
            
            # Create plot
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(features)), importances)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_n} Feature Importances - LightGBM')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Error plotting feature importance: {e}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        summary = self.get_model_info()
        
        if self.is_trained:
            summary.update({
                'model_type': 'LightGBM Classifier',
                'n_estimators': self.model.n_estimators,
                'num_leaves': self.model.num_leaves,
                'learning_rate': self.model.learning_rate,
                'detailed_feature_importance': self.get_detailed_feature_importance()
            })
        
        return summary
    
    def save_model_native(self, filename: str = None) -> str:
        """Save model in LightGBM native format"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        if filename is None:
            filename = f"{self.model_name}_native.txt"
        
        filepath = self.model_dir / filename
        self.model.booster_.save_model(str(filepath))
        
        logger.info(f"LightGBM native model saved to {filepath}")
        return str(filepath)
    
    def load_model_native(self, filepath: str) -> None:
        """Load model from LightGBM native format"""
        # Create a new model instance
        self.model = lgb.LGBMClassifier(**self.default_params)
        
        # Load the booster
        booster = lgb.Booster(model_file=filepath)
        self.model._Booster = booster
        
        self.is_trained = True
        logger.info(f"LightGBM native model loaded from {filepath}")
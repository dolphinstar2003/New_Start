"""
Base Model Class for ML Trading Models
Provides common interface and functionality
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class BaseMLModel(ABC):
    """Base class for all ML trading models"""
    
    def __init__(self, model_name: str, model_dir: Path):
        """
        Initialize base model
        
        Args:
            model_name: Name of the model
            model_dir: Directory to save models
        """
        self.model_name = model_name
        self.model_dir = model_dir
        self.model_dir.mkdir(exist_ok=True, parents=True)
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.is_trained = False
        
        # Model metadata
        self.training_stats = {}
        self.feature_importance = {}
        
        logger.info(f"Initialized {model_name} model")
    
    @abstractmethod
    def _create_model(self, **kwargs) -> Any:
        """Create the underlying ML model (to be implemented by subclasses)"""
        pass
    
    @abstractmethod
    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model (to be implemented by subclasses)"""
        pass
    
    @abstractmethod
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities (to be implemented by subclasses)"""
        pass
    
    def prepare_features(self, features: pd.DataFrame, target: pd.Series = None, 
                        fit_scalers: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for training/prediction
        
        Args:
            features: Feature DataFrame
            target: Target Series (optional)
            fit_scalers: Whether to fit scalers (True for training)
            
        Returns:
            Tuple of (X, y) arrays
        """
        # Handle missing values
        features_clean = features.fillna(method='ffill').fillna(0)
        
        # Store feature names
        if fit_scalers:
            self.feature_names = list(features_clean.columns)
        
        # Scale features
        if fit_scalers:
            X = self.scaler.fit_transform(features_clean)
        else:
            X = self.scaler.transform(features_clean)
        
        # Prepare target
        y = None
        if target is not None:
            if fit_scalers:
                # Fit label encoder for classification
                if target.dtype in ['object', 'category'] or target.nunique() <= 10:
                    y = self.label_encoder.fit_transform(target)
                else:
                    y = target.values
            else:
                # Transform using fitted encoder
                if hasattr(self.label_encoder, 'classes_'):
                    y = self.label_encoder.transform(target)
                else:
                    y = target.values
        
        return X, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   test_size: float = 0.2, time_series: bool = True) -> Tuple[np.ndarray, ...]:
        """
        Split data into train/test sets
        
        Args:
            X: Feature array
            y: Target array
            test_size: Fraction for test set
            time_series: Whether to use time series split
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        if time_series:
            # For time series, use last portion as test set
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
        else:
            # Random split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        return X_train, X_test, y_train, y_test
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      y_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # ROC AUC for binary/multiclass
        if y_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    # Binary classification
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    # Multiclass
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")
                metrics['roc_auc'] = 0.0
        
        # Trading-specific metrics
        if len(np.unique(y_true)) > 2:  # Multi-class
            # Calculate per-class precision/recall
            for i, class_name in enumerate(['Strong Down', 'Down', 'Hold', 'Up', 'Strong Up'][:len(np.unique(y_true))]):
                class_mask = (y_true == i)
                if class_mask.sum() > 0:
                    class_pred_mask = (y_pred == i)
                    if class_pred_mask.sum() > 0:
                        metrics[f'{class_name.lower()}_precision'] = (class_mask & class_pred_mask).sum() / class_pred_mask.sum()
                    else:
                        metrics[f'{class_name.lower()}_precision'] = 0.0
                    metrics[f'{class_name.lower()}_recall'] = (class_mask & (y_pred == i)).sum() / class_mask.sum()
        
        return metrics
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance (to be overridden by specific models)"""
        return {}
    
    def train(self, features: pd.DataFrame, target: pd.Series, 
              validation_split: float = 0.2, **kwargs) -> Dict[str, Any]:
        """
        Train the model
        
        Args:
            features: Training features
            target: Training target
            validation_split: Validation split ratio
            **kwargs: Additional model parameters
            
        Returns:
            Training statistics
        """
        logger.info(f"Training {self.model_name} model...")
        
        # Prepare data
        X, y = self.prepare_features(features, target, fit_scalers=True)
        
        # Split data
        X_train, X_val, y_train, y_val = self.split_data(X, y, test_size=validation_split)
        
        # Create and train model
        self.model = self._create_model(**kwargs)
        self._fit_model(X_train, y_train)
        
        # Evaluate on validation set
        y_val_pred = self.predict(features.iloc[len(X_train):])
        y_val_proba = self._predict_proba(X_val)
        
        # Calculate metrics
        val_metrics = self.evaluate_model(y_val, y_val_pred, y_val_proba)
        
        # Store training statistics
        self.training_stats = {
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'features_count': len(self.feature_names),
            'validation_metrics': val_metrics
        }
        
        # Get feature importance
        self.feature_importance = self.get_feature_importance()
        
        self.is_trained = True
        logger.info(f"Training completed. Validation accuracy: {val_metrics['accuracy']:.4f}")
        
        return self.training_stats
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Make predictions
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X, _ = self.prepare_features(features, fit_scalers=False)
        predictions = self.model.predict(X)
        
        # Convert back from label encoding if necessary
        if hasattr(self.label_encoder, 'classes_'):
            predictions = self.label_encoder.inverse_transform(predictions)
        
        return predictions
    
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Predicted probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X, _ = self.prepare_features(features, fit_scalers=False)
        return self._predict_proba(X)
    
    def save_model(self, filename: str = None) -> Path:
        """
        Save trained model
        
        Args:
            filename: Optional filename (uses model_name if not provided)
            
        Returns:
            Path to saved model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        if filename is None:
            filename = f"{self.model_name}_model.joblib"
        
        filepath = self.model_dir / filename
        
        # Save model components
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'training_stats': self.training_stats,
            'feature_importance': self.feature_importance,
            'model_name': self.model_name
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
        
        return filepath
    
    def load_model(self, filepath: Path) -> None:
        """
        Load trained model
        
        Args:
            filepath: Path to saved model
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.training_stats = model_data['training_stats']
        self.feature_importance = model_data['feature_importance']
        
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        info = {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_names),
            'training_stats': self.training_stats
        }
        
        if self.feature_importance:
            # Top 10 most important features
            sorted_features = sorted(self.feature_importance.items(), 
                                   key=lambda x: x[1], reverse=True)
            info['top_features'] = dict(sorted_features[:10])
        
        return info
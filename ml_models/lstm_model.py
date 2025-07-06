"""
LSTM Model for Trading Signal Prediction
CPU-only TensorFlow implementation following God Mode rules
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

try:
    import tensorflow as tf
    # Force CPU-only usage (God Mode requirement)
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)  # Limit CPU usage
    tf.config.threading.set_inter_op_parallelism_threads(4)
    
    from tensorflow import keras
    from tensorflow.keras import layers, Model, callbacks
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    keras = None
    layers = None
    Model = None
    callbacks = None

from .base_model import BaseMLModel
from loguru import logger


class LSTMTradingModel(BaseMLModel):
    """LSTM model for trading signal prediction"""
    
    def __init__(self, model_dir):
        """Initialize LSTM model"""
        super().__init__("LSTM", model_dir)
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available. Install with: pip install tensorflow")
        
        # LSTM-specific parameters
        self.sequence_length = 20  # Look back 20 days
        self.default_params = {
            'lstm_units': [64, 32],  # Two LSTM layers
            'dropout_rate': 0.2,
            'recurrent_dropout': 0.2,
            'dense_units': [32, 16],  # Dense layers after LSTM
            'activation': 'relu',
            'output_activation': 'softmax',
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 50,
            'validation_split': 0.2,
            'early_stopping_patience': 10,
            'num_classes': 3  # -1: sell, 0: hold, 1: buy
        }
        
        # Store sequences for prediction
        self.last_sequence = None
        self.sequence_scaler = None
        
        logger.info("LSTM model initialized (CPU-only)")
    
    def _create_model(self, **kwargs) -> Model:
        """Create LSTM model architecture"""
        params = {**self.default_params, **kwargs}
        
        logger.info(f"Creating LSTM model with params: {params}")
        
        # Input layer
        inputs = keras.Input(shape=(self.sequence_length, len(self.feature_names)))
        
        # LSTM layers
        x = inputs
        for i, units in enumerate(params['lstm_units']):
            return_sequences = i < len(params['lstm_units']) - 1
            x = layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=params['dropout_rate'],
                recurrent_dropout=params['recurrent_dropout'],
                name=f'lstm_{i+1}'
            )(x)
        
        # Dense layers
        for i, units in enumerate(params['dense_units']):
            x = layers.Dense(
                units,
                activation=params['activation'],
                name=f'dense_{i+1}'
            )(x)
            x = layers.Dropout(params['dropout_rate'])(x)
        
        # Output layer
        outputs = layers.Dense(
            params['num_classes'],
            activation=params['output_activation'],
            name='output'
        )(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='LSTM_Trading_Model')
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=params['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'sparse_categorical_crossentropy']
        )
        
        return model
    
    def _prepare_sequences(self, features: pd.DataFrame, target: pd.Series = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training/prediction
        
        Args:
            features: Feature DataFrame
            target: Target Series (optional)
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        # Convert to numpy
        feature_array = features.values
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(self.sequence_length, len(feature_array)):
            # Feature sequence
            X_sequences.append(feature_array[i-self.sequence_length:i])
            
            # Target (if provided)
            if target is not None:
                y_sequences.append(target.iloc[i])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences) if y_sequences else None
        
        return X_sequences, y_sequences
    
    def prepare_features(self, features: pd.DataFrame, target: pd.Series = None, 
                        fit_scalers: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for LSTM (override base method)
        
        Args:
            features: Feature DataFrame
            target: Target Series (optional)
            fit_scalers: Whether to fit scalers
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        # Handle missing values
        features_clean = features.fillna(method='ffill').fillna(0)
        
        # Store feature names
        if fit_scalers:
            self.feature_names = list(features_clean.columns)
        
        # Scale features using parent method
        X_scaled, y = super().prepare_features(features_clean, target, fit_scalers)
        
        # Convert back to DataFrame for sequence preparation
        features_scaled = pd.DataFrame(X_scaled, index=features_clean.index, columns=self.feature_names)
        target_series = pd.Series(y, index=features_clean.index) if y is not None else None
        
        # Create sequences
        X_sequences, y_sequences = self._prepare_sequences(features_scaled, target_series)
        
        return X_sequences, y_sequences
    
    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit LSTM model"""
        logger.info(f"Training LSTM on {X.shape[0]} sequences of length {X.shape[1]} with {X.shape[2]} features")
        
        # Prepare callbacks
        callbacks_list = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.default_params['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        history = self.model.fit(
            X, y,
            batch_size=self.default_params['batch_size'],
            epochs=self.default_params['epochs'],
            validation_split=self.default_params['validation_split'],
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Store training history
        self.training_history = history.history
        
        logger.info("LSTM training completed")
    
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        return self.model.predict(X, verbose=0)
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Make predictions (override to handle sequences)
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_sequences, _ = self.prepare_features(features, fit_scalers=False)
        
        if len(X_sequences) == 0:
            logger.warning("Not enough data for sequence prediction")
            return np.array([])
        
        # Get probabilities
        probabilities = self._predict_proba(X_sequences)
        
        # Convert to class predictions
        predictions = np.argmax(probabilities, axis=1)
        
        # Convert back from label encoding if necessary
        if hasattr(self.label_encoder, 'classes_'):
            predictions = self.label_encoder.inverse_transform(predictions)
        
        return predictions
    
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities (override to handle sequences)
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Predicted probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_sequences, _ = self.prepare_features(features, fit_scalers=False)
        
        if len(X_sequences) == 0:
            logger.warning("Not enough data for sequence prediction")
            return np.array([])
        
        return self._predict_proba(X_sequences)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance using permutation importance
        Note: LSTM doesn't have built-in feature importance like tree models
        """
        if not self.is_trained:
            return {}
        
        logger.info("Calculating permutation importance for LSTM (this may take a while)")
        
        try:
            from sklearn.inspection import permutation_importance
            
            # We need validation data for permutation importance
            # This is a simplified version - in practice you'd use held-out validation set
            importance_dict = {}
            
            # For now, return empty dict as LSTM feature importance is complex
            logger.warning("LSTM feature importance calculation not implemented yet")
            return importance_dict
            
        except ImportError:
            logger.warning("Scikit-learn not available for permutation importance")
            return {}
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return {}
    
    def plot_training_history(self) -> None:
        """Plot training history"""
        try:
            import matplotlib.pyplot as plt
            
            if not hasattr(self, 'training_history'):
                logger.warning("No training history available")
                return
            
            history = self.training_history
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot training & validation accuracy
            ax1.plot(history['accuracy'], label='Training Accuracy')
            ax1.plot(history['val_accuracy'], label='Validation Accuracy')
            ax1.set_title('Model Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            
            # Plot training & validation loss
            ax2.plot(history['loss'], label='Training Loss')
            ax2.plot(history['val_loss'], label='Validation Loss')
            ax2.set_title('Model Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Error plotting training history: {e}")
    
    def save_model(self, filename: str = None) -> Path:
        """Save LSTM model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        if filename is None:
            filename = f"{self.model_name}_model"
        
        # Save TensorFlow model
        model_path = self.model_dir / filename
        self.model.save(model_path, save_format='tf')
        
        # Save additional components using joblib
        import joblib
        metadata_path = self.model_dir / f"{filename}_metadata.joblib"
        
        metadata = {
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'training_stats': self.training_stats,
            'sequence_length': self.sequence_length,
            'model_name': self.model_name
        }
        
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"LSTM model saved to {model_path}")
        return model_path
    
    def load_model(self, filepath: Path) -> None:
        """Load LSTM model"""
        import joblib
        
        # Load TensorFlow model
        if filepath.is_dir():
            self.model = keras.models.load_model(filepath)
        else:
            # Assume it's the metadata file, derive model path
            model_path = filepath.parent / filepath.stem.replace('_metadata', '')
            if model_path.exists():
                self.model = keras.models.load_model(model_path)
            else:
                raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        # Load metadata
        if str(filepath).endswith('_metadata.joblib'):
            metadata_path = filepath
        else:
            metadata_path = filepath.parent / f"{filepath.name}_metadata.joblib"
        
        if metadata_path.exists():
            metadata = joblib.load(metadata_path)
            self.scaler = metadata['scaler']
            self.label_encoder = metadata['label_encoder']
            self.feature_names = metadata['feature_names']
            self.training_stats = metadata.get('training_stats', {})
            self.sequence_length = metadata.get('sequence_length', 20)
        
        self.is_trained = True
        logger.info(f"LSTM model loaded from {filepath}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        summary = self.get_model_info()
        
        if self.is_trained:
            summary.update({
                'model_type': 'LSTM Neural Network',
                'sequence_length': self.sequence_length,
                'total_params': self.model.count_params(),
                'architecture': str(self.model.summary())
            })
            
            if hasattr(self, 'training_history'):
                # Get best metrics from training
                history = self.training_history
                summary.update({
                    'best_val_accuracy': max(history.get('val_accuracy', [0])),
                    'best_val_loss': min(history.get('val_loss', [float('inf')])),
                    'training_epochs': len(history.get('loss', []))
                })
        
        return summary
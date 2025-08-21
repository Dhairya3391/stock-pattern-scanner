"""
Model Training Script for Advanced Pattern Scanner.

This script trains the CNN-LSTM model using synthetic data generated
from reference algorithm specifications. It creates a lightweight model
optimized for Apple Silicon performance.
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Dict, List
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

from .model_manager import PatternCNNLSTM
from .synthetic_data_generator import SyntheticDataGenerator
from ..core.models import PatternConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)


class ModelTrainer:
    """
    Trains ML models for pattern detection and validation.
    
    This class handles the complete training pipeline including data generation,
    model training, validation, and saving of trained models.
    """
    
    def __init__(self, config: PatternConfig):
        """
        Initialize the model trainer.
        
        Args:
            config: Pattern detection configuration
        """
        self.config = config
        self.device = self._get_optimal_device()
        
        # Training parameters
        self.batch_size = 32
        self.learning_rate = 0.001
        self.num_epochs = 50
        self.early_stopping_patience = 10
        
        # Data generator
        self.data_generator = SyntheticDataGenerator()
        
        logger.info(f"ModelTrainer initialized with device: {self.device}")
    
    def _get_optimal_device(self) -> torch.device:
        """Get optimal device for training."""
        if torch.backends.mps.is_available() and self.config.use_gpu:
            return torch.device("mps")
        elif torch.cuda.is_available() and self.config.use_gpu:
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def train_all_models(self, num_samples: int = 5000, 
                        model_dir: str = "models") -> Dict[str, bool]:
        """
        Train all models (CNN-LSTM and fallback).
        
        Args:
            num_samples: Number of synthetic samples to generate
            model_dir: Directory to save trained models
            
        Returns:
            Dictionary indicating success of each model training
        """
        results = {}
        
        # Create model directory
        model_path = Path(model_dir)
        model_path.mkdir(exist_ok=True)
        
        logger.info(f"Starting training with {num_samples} samples")
        
        # Generate synthetic dataset
        logger.info("Generating synthetic dataset...")
        features, labels, pattern_names = self.data_generator.generate_dataset(
            num_samples=num_samples,
            sequence_length=120
        )
        
        # Train CNN-LSTM model
        logger.info("Training CNN-LSTM model...")
        try:
            cnn_lstm_success = self._train_cnn_lstm(
                features, labels, pattern_names, model_path
            )
            results["cnn_lstm"] = cnn_lstm_success
        except Exception as e:
            logger.error(f"CNN-LSTM training failed: {e}")
            results["cnn_lstm"] = False
        
        # Train fallback model
        logger.info("Training fallback Random Forest model...")
        try:
            fallback_success = self._train_fallback_model(
                features, labels, pattern_names, model_path
            )
            results["fallback"] = fallback_success
        except Exception as e:
            logger.error(f"Fallback model training failed: {e}")
            results["fallback"] = False
        
        logger.info(f"Training completed. Results: {results}")
        return results
    
    def _train_cnn_lstm(self, features: np.ndarray, labels: np.ndarray, 
                       pattern_names: List[str], model_path: Path) -> bool:
        """
        Train the CNN-LSTM model.
        
        Args:
            features: Feature array
            labels: Label array
            pattern_names: List of pattern names
            model_path: Path to save model
            
        Returns:
            True if training successful
        """
        try:
            # Reshape features for CNN-LSTM (samples, sequence_length, features)
            # Features are flattened (60 * 10), need to reshape
            sequence_length = 60
            num_features = 10
            
            # Ensure features can be reshaped
            if features.shape[1] != sequence_length * num_features:
                logger.warning(f"Feature shape mismatch: {features.shape[1]} != {sequence_length * num_features}")
                # Pad or truncate as needed
                target_size = sequence_length * num_features
                if features.shape[1] < target_size:
                    padding = np.zeros((features.shape[0], target_size - features.shape[1]))
                    features = np.concatenate([features, padding], axis=1)
                else:
                    features = features[:, :target_size]
            
            # Reshape to (samples, sequence_length, num_features)
            features_reshaped = features.reshape(-1, sequence_length, num_features)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                features_reshaped, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.LongTensor(y_train)
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.LongTensor(y_val)
            
            # Create data loaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            
            # Initialize model
            model = PatternCNNLSTM(
                input_features=num_features,
                sequence_length=sequence_length,
                num_classes=len(pattern_names)
            )
            model.to(self.device)
            
            # Loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            train_losses = []
            val_losses = []
            
            for epoch in range(self.num_epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for batch_features, batch_labels in train_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    class_logits, confidence = model(batch_features)
                    loss = criterion(class_logits, batch_labels)
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = torch.max(class_logits.data, 1)
                    train_total += batch_labels.size(0)
                    train_correct += (predicted == batch_labels).sum().item()
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_features, batch_labels in val_loader:
                        batch_features = batch_features.to(self.device)
                        batch_labels = batch_labels.to(self.device)
                        
                        class_logits, confidence = model(batch_features)
                        loss = criterion(class_logits, batch_labels)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(class_logits.data, 1)
                        val_total += batch_labels.size(0)
                        val_correct += (predicted == batch_labels).sum().item()
                
                # Calculate averages
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                train_acc = 100 * train_correct / train_total
                val_acc = 100 * val_correct / val_total
                
                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)
                
                # Learning rate scheduling
                scheduler.step(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    
                    # Save best model
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': avg_train_loss,
                        'val_loss': avg_val_loss,
                        'input_features': num_features,
                        'sequence_length': sequence_length,
                        'num_classes': len(pattern_names),
                        'pattern_names': pattern_names
                    }, model_path / "pattern_cnn_lstm.pth")
                    
                else:
                    patience_counter += 1
                
                # Log progress
                if epoch % 10 == 0 or epoch == self.num_epochs - 1:
                    logger.info(f"Epoch {epoch+1}/{self.num_epochs}: "
                              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                
                # Early stopping
                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            logger.info("CNN-LSTM training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"CNN-LSTM training failed: {e}")
            return False
    
    def _train_fallback_model(self, features: np.ndarray, labels: np.ndarray, 
                            pattern_names: List[str], model_path: Path) -> bool:
        """
        Train the fallback Random Forest model.
        
        Args:
            features: Feature array
            labels: Label array
            pattern_names: List of pattern names
            model_path: Path to save model
            
        Returns:
            True if training successful
        """
        try:
            # For Random Forest, we can use flattened features
            # But let's reduce dimensionality for better performance
            
            # Feature selection: take every 10th feature to reduce from 600 to 60
            selected_features = features[:, ::10]
            
            # Ensure we have exactly 50 features (pad if necessary)
            if selected_features.shape[1] < 50:
                padding = np.zeros((selected_features.shape[0], 50 - selected_features.shape[1]))
                selected_features = np.concatenate([selected_features, padding], axis=1)
            elif selected_features.shape[1] > 50:
                selected_features = selected_features[:, :50]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                selected_features, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Random Forest
            rf_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1  # Use all available cores
            )
            
            logger.info("Training Random Forest model...")
            rf_model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_score = rf_model.score(X_train_scaled, y_train)
            test_score = rf_model.score(X_test_scaled, y_test)
            
            logger.info(f"Random Forest - Train Accuracy: {train_score:.4f}, Test Accuracy: {test_score:.4f}")
            
            # Generate classification report
            y_pred = rf_model.predict(X_test_scaled)
            report = classification_report(y_test, y_pred, target_names=pattern_names)
            logger.info(f"Classification Report:\n{report}")
            
            # Save model and scaler
            joblib.dump(rf_model, model_path / "pattern_random_forest.joblib")
            joblib.dump(scaler, model_path / "feature_scaler.joblib")
            
            # Save feature importance
            feature_importance = rf_model.feature_importances_
            np.save(model_path / "feature_importance.npy", feature_importance)
            
            logger.info("Random Forest training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Random Forest training failed: {e}")
            return False
    
    def evaluate_models(self, model_dir: str = "models") -> Dict[str, Dict]:
        """
        Evaluate trained models on test data.
        
        Args:
            model_dir: Directory containing trained models
            
        Returns:
            Evaluation results for each model
        """
        results = {}
        
        # Generate test data
        logger.info("Generating test dataset for evaluation...")
        test_features, test_labels, pattern_names = self.data_generator.generate_dataset(
            num_samples=1000,
            sequence_length=120
        )
        
        model_path = Path(model_dir)
        
        # Evaluate CNN-LSTM
        cnn_lstm_path = model_path / "pattern_cnn_lstm.pth"
        if cnn_lstm_path.exists():
            try:
                results["cnn_lstm"] = self._evaluate_cnn_lstm(
                    test_features, test_labels, pattern_names, cnn_lstm_path
                )
            except Exception as e:
                logger.error(f"CNN-LSTM evaluation failed: {e}")
                results["cnn_lstm"] = {"error": str(e)}
        
        # Evaluate Random Forest
        rf_path = model_path / "pattern_random_forest.joblib"
        scaler_path = model_path / "feature_scaler.joblib"
        
        if rf_path.exists() and scaler_path.exists():
            try:
                results["random_forest"] = self._evaluate_random_forest(
                    test_features, test_labels, pattern_names, rf_path, scaler_path
                )
            except Exception as e:
                logger.error(f"Random Forest evaluation failed: {e}")
                results["random_forest"] = {"error": str(e)}
        
        return results
    
    def _evaluate_cnn_lstm(self, features: np.ndarray, labels: np.ndarray, 
                          pattern_names: List[str], model_path: Path) -> Dict:
        """Evaluate CNN-LSTM model."""
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        
        model = PatternCNNLSTM(
            input_features=checkpoint['input_features'],
            sequence_length=checkpoint['sequence_length'],
            num_classes=checkpoint['num_classes']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        # Prepare data
        sequence_length = checkpoint['sequence_length']
        num_features = checkpoint['input_features']
        
        if features.shape[1] != sequence_length * num_features:
            target_size = sequence_length * num_features
            if features.shape[1] < target_size:
                padding = np.zeros((features.shape[0], target_size - features.shape[1]))
                features = np.concatenate([features, padding], axis=1)
            else:
                features = features[:, :target_size]
        
        features_reshaped = features.reshape(-1, sequence_length, num_features)
        X_tensor = torch.FloatTensor(features_reshaped).to(self.device)
        
        # Evaluate
        predictions = []
        confidences = []
        
        with torch.no_grad():
            for i in range(0, len(X_tensor), self.batch_size):
                batch = X_tensor[i:i+self.batch_size]
                class_logits, confidence = model(batch)
                
                _, predicted = torch.max(class_logits, 1)
                predictions.extend(predicted.cpu().numpy())
                confidences.extend(confidence.cpu().numpy())
        
        # Calculate metrics
        accuracy = np.mean(np.array(predictions) == labels)
        avg_confidence = np.mean(confidences)
        
        return {
            "accuracy": accuracy,
            "average_confidence": avg_confidence,
            "num_samples": len(labels)
        }
    
    def _evaluate_random_forest(self, features: np.ndarray, labels: np.ndarray, 
                              pattern_names: List[str], model_path: Path, 
                              scaler_path: Path) -> Dict:
        """Evaluate Random Forest model."""
        # Load model and scaler
        rf_model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Prepare features
        selected_features = features[:, ::10]
        if selected_features.shape[1] < 50:
            padding = np.zeros((selected_features.shape[0], 50 - selected_features.shape[1]))
            selected_features = np.concatenate([selected_features, padding], axis=1)
        elif selected_features.shape[1] > 50:
            selected_features = selected_features[:, :50]
        
        # Scale features
        features_scaled = scaler.transform(selected_features)
        
        # Evaluate
        accuracy = rf_model.score(features_scaled, labels)
        predictions = rf_model.predict(features_scaled)
        probabilities = rf_model.predict_proba(features_scaled)
        avg_confidence = np.mean(np.max(probabilities, axis=1))
        
        return {
            "accuracy": accuracy,
            "average_confidence": avg_confidence,
            "num_samples": len(labels)
        }


def main():
    """Main training function."""
    # Initialize configuration
    config = PatternConfig()
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    # Train models
    logger.info("Starting model training...")
    results = trainer.train_all_models(num_samples=5000)
    
    # Evaluate models
    if any(results.values()):
        logger.info("Evaluating trained models...")
        eval_results = trainer.evaluate_models()
        
        for model_name, metrics in eval_results.items():
            if "error" not in metrics:
                logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, "
                          f"Avg Confidence: {metrics['average_confidence']:.4f}")
    
    logger.info("Training pipeline completed!")


if __name__ == "__main__":
    main()
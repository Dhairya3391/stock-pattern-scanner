"""
Demo script for training ML models.

This script demonstrates how to train the CNN-LSTM and Random Forest models
using synthetic data generated from reference algorithm specifications.
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from advanced_pattern_scanner.core.models import PatternConfig
from advanced_pattern_scanner.ml import ModelTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main demo function."""
    logger.info("Starting ML model training demo...")
    
    # Initialize configuration
    config = PatternConfig()
    
    # Create trainer
    trainer = ModelTrainer(config)
    
    # Train models with a smaller dataset for demo
    logger.info("Training models with synthetic data...")
    results = trainer.train_all_models(num_samples=1000)  # Smaller dataset for demo
    
    # Report results
    logger.info("Training Results:")
    for model_name, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"  {model_name}: {status}")
    
    # Evaluate models if training was successful
    if any(results.values()):
        logger.info("Evaluating trained models...")
        eval_results = trainer.evaluate_models()
        
        logger.info("Evaluation Results:")
        for model_name, metrics in eval_results.items():
            if "error" not in metrics:
                logger.info(f"  {model_name}:")
                logger.info(f"    Accuracy: {metrics['accuracy']:.4f}")
                logger.info(f"    Avg Confidence: {metrics['average_confidence']:.4f}")
                logger.info(f"    Samples: {metrics['num_samples']}")
            else:
                logger.error(f"  {model_name}: {metrics['error']}")
    
    logger.info("Demo completed!")


if __name__ == "__main__":
    main()
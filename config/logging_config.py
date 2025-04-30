import logging
import sys
import os

def setup_training_logger(output_dir):
    """Setup logging configuration for training"""
    logger = logging.getLogger("CogmentaTrainer")
    logger.setLevel(logging.INFO)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # File handler with UTF-8 encoding
    file_handler = logging.FileHandler(
        os.path.join(output_dir, "training.log"),
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    
    # Console handler with UTF-8 encoding
    if sys.platform == 'win32':
        # Windows-specific handling
        sys.stdout.reconfigure(encoding='utf-8')
        console_handler = logging.StreamHandler(sys.stdout)
    else:
        console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

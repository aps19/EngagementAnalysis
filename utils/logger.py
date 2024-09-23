import logging
import sys
import os
from logging.handlers import RotatingFileHandler

# Setup the logger
def setup_logging(log_dir='log_dir', log_filename='training.log'):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)
    
    # Setup logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()  # Log to console as well
        ]
    )
    logger = logging.getLogger()
    return logger


import logging
import os
import sys
from colorama import Fore, Style, init
from .config import Config

# Initialize colorama
init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    """
    Custom formatter to inject colors into terminal logs based on severity level.
    """
    COLORS = {
        logging.DEBUG: Fore.CYAN + Style.DIM,
        logging.INFO: Fore.WHITE,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        original_msg = record.msg
        
        # Apply color only for terminal output
        if record.levelno in self.COLORS:
            # Αν το μήνυμα έχει ήδη χρώμα (π.χ. από το merger.py), δεν πειράζουμε τίποτα
            if isinstance(original_msg, str) and "\033[" in original_msg:
                pass 
            else:
                record.msg = self.COLORS[record.levelno] + str(original_msg) + Style.RESET_ALL
        
        result = super().format(record)
        record.msg = original_msg
        return result

def setup_logger(name="ThesisLogger"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG) 
    
    # --- ΣΗΜΑΝΤΙΚΗ ΑΛΛΑΓΗ ---
    # Αποτρέπει την αποστολή των logs στον root logger (σταματάει τα διπλότυπα)
    logger.propagate = False 
    # ------------------------

    if logger.hasHandlers():
        logger.handlers.clear()

    # Handler 1: File (Detailed)
    log_file = os.path.join(Config.LOGS_DIR, 'app_debug.log')
    # Δημιουργία φακέλου logs αν δεν υπάρχει
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter('%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s')
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    # Handler 2: Console (Clean)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_fmt = ColoredFormatter('%(message)s')
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    return logger

log = setup_logger()
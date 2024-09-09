from pathlib import Path
import yaml
import logging

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# function to get the parameters

def get_parameters(param=None):
    if param is None:
        raise ValueError("The parameter cannot be None.")
    with open(PROJ_ROOT / "params.yaml",'r') as f:
        return yaml.safe_load(f.read())[param]

def create_logger(level='DEBUG', file_name ="no file") -> logging.Logger:
    # configure logger
    logger = logging.getLogger(file_name)
    logger.setLevel('DEBUG')

    console_handler   = logging.StreamHandler()
    console_handler.setLevel('DEBUG')

    console_handler.setFormatter(logging.Formatter('%(asctime)s %(name)s - %(levelname)s %(message)s'))
    
    logger.addHandler(console_handler)
    return logger

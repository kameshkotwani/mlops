from pathlib import Path
import yaml

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
print(f"PROJ_ROOT path is: {PROJ_ROOT}")

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


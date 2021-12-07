import os
from pathlib import Path

CONSTANTS_FILE_LOCATION = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = Path(CONSTANTS_FILE_LOCATION).parent
CONFIG_FILE_NAME = "default-config.cfg"
CONFIG_FILE_PATH = os.path.join(PROJECT_ROOT,CONFIG_FILE_NAME)
BODY_FILE_PATH = os.path.join(PROJECT_ROOT,"default-bodies.cfg")
BODY_FILE_NAME = "default-bodies.cfg"
OUTPUT_DIR_LOCATION = os.path.join(PROJECT_ROOT,"outputs")
SAVED_RUNS_DIR = os.path.join(PROJECT_ROOT,"saved-runs")
PRESETS_DIR = os.path.join(PROJECT_ROOT,"bodies-presets")

VELOCITY_SCALER = 3 # Historically was 36


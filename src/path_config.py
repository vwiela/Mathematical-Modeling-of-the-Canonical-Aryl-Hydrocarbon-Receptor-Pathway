from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PETAB_DIR = PROJECT_ROOT / "petab"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_SELECTION_RESULTS_DIR = RESULTS_DIR / "model_selection"
FIGURES_DIR = RESULTS_DIR / "figures"

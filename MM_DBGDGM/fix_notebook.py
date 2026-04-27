"""Fix all issues in full_pipeline_demo.ipynb"""
import json

path = "notebooks/full_pipeline_demo.ipynb"

with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# ============================================================
# Fix 1: Cell 0 (setup cell) - fix path handling
# The notebook runs from notebooks/, but all paths assume root.
# Add os.chdir to go to root, so DATA_DIR and OUT_DIR work.
# Also ensure results/figures/ exists.
# ============================================================
for i, cell in enumerate(cells):
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if "DATA_DIR" in src and "OUT_DIR" in src and "sys.path.insert" in src:
            cell["source"] = [
                "import sys, os\n",
                "\n",
                "# Change to project root so all relative paths work\n",
                "os.chdir(os.path.join(os.path.dirname(os.path.abspath('.')), ''))\n",
                "# Ensure project root is on the Python path\n",
                "if os.getcwd() not in sys.path:\n",
                "    sys.path.insert(0, os.getcwd())\n",
                "\n",
                "import numpy as np\n",
                "import torch\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from pathlib import Path\n",
                "\n",
                "SEED = 42\n",
                "np.random.seed(SEED)\n",
                "torch.manual_seed(SEED)\n",
                "\n",
                "DATA_DIR = Path(\"data/synthetic/subjects\")\n",
                "OUT_DIR  = Path(\"results\")\n",
                "\n",
                "# Ensure output directories exist\n",
                "(OUT_DIR / \"figures\").mkdir(parents=True, exist_ok=True)\n",
            ]
            cell["outputs"] = []
            cell["execution_count"] = None
            print(f"  Fixed cell {i}: setup cell (path handling + mkdir)")
            break

# ============================================================
# Fix 2: Training cell - fix config.yaml path
# Set config to None so run_experiment uses defaults.
# ============================================================
for i, cell in enumerate(cells):
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if '"config":' in src and "run_experiment" in src:
            cell["source"] = [
                "import yaml\n",
                "from run_experiment import run\n",
                "\n",
                "# Build args (no config file needed — defaults are fine)\n",
                "args_dict = {\n",
                '    "data_dir":   str(DATA_DIR),\n',
                '    "manifest":   str(DATA_DIR / "manifest.csv"),\n',
                '    "epochs":     50,   # reduced for demo speed\n',
                '    "beta":       2.0,\n',
                '    "seed":       SEED,\n',
                '    "k_folds":    5,\n',
                '    "batch_size": 8,\n',
                '    "output_dir": str(OUT_DIR),\n',
                '    "config":     None,  # use defaults, no config file needed\n',
                "}\n",
                "import argparse\n",
                "args = argparse.Namespace(**args_dict)\n",
                "\n",
                'print("Starting 5-fold CV training (50 epochs/fold)...")\n',
                "summary = run(args)\n",
            ]
            cell["outputs"] = []
            cell["execution_count"] = None
            print(f"  Fixed cell {i}: training cell (config=None)")
            break

# ============================================================
# Fix 3: Clear all outputs and execution counts
# so the notebook starts clean.
# ============================================================
for i, cell in enumerate(cells):
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

# Validate
try:
    json.dumps(nb)
    print("  JSON validation passed!")
except Exception as e:
    print(f"  JSON validation failed: {e}")

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("  Notebook fixed and saved.")

"""
Launcher for the synthetic data engine with file/folder selection dialogs.

Lets you select one or more reference CSV files and an output folder by clicking
instead of typing paths. Then runs the synthetic dataset builder for each
selected file (each run saves to a subfolder under the output folder so they
don't overwrite each other).

- One file selected: output goes to <output_folder>/run_<filename_stem>/
- Multiple files: each run goes to <output_folder>/run_<filename_stem>/ so
  results don't overwrite.

Uses defaults: 1000 rows, base_case scenario, no synthetic outcomes.
For full options (scenario, outcomes, target bad rate, etc.) run:
  python build_synthetic_dataset.py --input <path> --output-dir <path> ...

Requires: tkinter (usually bundled with Python on Windows).
"""

import os
import subprocess
import sys
from pathlib import Path

# Try tkinter for file/folder dialogs
try:
    import tkinter as tk
    from tkinter import filedialog
    HAS_TK = True
except ImportError:
    HAS_TK = False

# Defaults
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data" / "synthetic"
DEFAULT_ROWS = 1000
DEFAULT_SCENARIO = "base_case"
DEFAULT_SEED = 42


def pick_files_and_folder():
    """Show dialogs to pick input CSV(s) and output folder. Returns (list of paths, output_dir) or (None, None) if cancelled."""
    if not HAS_TK:
        print("tkinter not available. Install it or run build_synthetic_dataset.py from the command line with --input and --output-dir.")
        return None, None

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    print("Select one or more reference CSV files (Ctrl+Click for multiple)...")
    input_paths = filedialog.askopenfilenames(
        title="Select reference CSV file(s)",
        initialdir=SCRIPT_DIR,
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    if not input_paths:
        print("No files selected. Exiting.")
        root.destroy()
        return None, None

    print("Select output folder for synthetic outputs...")
    output_dir = filedialog.askdirectory(
        title="Select output folder",
        initialdir=str(DEFAULT_OUTPUT_DIR) if DEFAULT_OUTPUT_DIR.exists() else str(SCRIPT_DIR),
    )
    if not output_dir:
        print("No output folder selected. Exiting.")
        root.destroy()
        return None, None

    root.destroy()
    return list(input_paths), output_dir


def run_builder(input_path: str, output_dir: str, subfolder: bool = True) -> int:
    """Run build_synthetic_dataset.py for one input file. Returns exit code."""
    if subfolder:
        stem = Path(input_path).stem
        out = os.path.join(output_dir, f"run_{stem}")
    else:
        out = output_dir
    os.makedirs(out, exist_ok=True)
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "build_synthetic_dataset.py"),
        "--input", input_path,
        "--output-dir", out,
        "--rows", str(DEFAULT_ROWS),
        "--scenario", DEFAULT_SCENARIO,
        "--seed", str(DEFAULT_SEED),
    ]
    return subprocess.call(cmd)


def main():
    print("\n=== Synthetic Data Engine - File Launcher ===\n")
    input_paths, output_dir = pick_files_and_folder()
    if not input_paths:
        return 1

    n = len(input_paths)
    for i, path in enumerate(input_paths, 1):
        print(f"\n--- Run {i}/{n}: {path} ---")
        # Multiple files: save each to output_dir/run_<stem>/ so they don't overwrite
        code = run_builder(path, output_dir, subfolder=(n > 1))
        if code != 0:
            print(f"Run failed with exit code {code}.")
            return code

    print("\nDone. Output(s) under:", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

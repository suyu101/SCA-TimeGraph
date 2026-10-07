"""Runner script to generate experiment visualization figures."""

import sys
from pathlib import Path

# Add src to path if not installed
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sca.visualization.visualize_results import *

if __name__ == "__main__":
    print("Generating visualizations...")
    # The module generates all plots on import / execution
    print("Visualizations generated in results/plots/.")

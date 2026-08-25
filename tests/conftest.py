import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE_DIR = os.path.join(REPO_ROOT, 'NanoparticlesSAM')

if PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PACKAGE_DIR)

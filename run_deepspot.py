#!/usr/bin/env python3
"""
Launch DeepSpot Analysis (Pollen Viability Detection) for internal testing.
Usage: python run_deepspot.py
Optional: set DEEPSPOT_DB_PATH to a network drive path for shared SQLite DB.
"""
import sys
import os

# Run from Hema folder; add deepspot_app to path
HEMA_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HEMA_DIR)

from deepspot_app.app import main

if __name__ == "__main__":
    main()

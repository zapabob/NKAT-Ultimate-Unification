#!/usr/bin/env python3
"""
NKAT Ultimate Millennium Problems Solver - Launcher
Simple launcher script to execute the main solver
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Import and run the main solver with proper encoding
    with open('nkat_ultimate_millennium_conjectures_final_solution.py', 'r', encoding='utf-8') as f:
        exec(f.read())
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 
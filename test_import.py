#!/usr/bin/env python3
"""Test import paths"""

import importlib
import sys

# Print the Python path
print("Python path:", sys.path)

# Try to import various modules
modules_to_check = [
    "backend",
    "backend.app",
    "backend.app.adapters",
    "backend.app.api",
    "backend.app.core",
    "backend.app.domain",
    "backend.app.ports",
    "backend.app.services",
]

for module_name in modules_to_check:
    try:
        module = importlib.import_module(module_name)
        print(f"✅ Successfully imported {module_name}")
    except ImportError as e:
        print(f"❌ Failed to import {module_name}: {e}")

print("\nTest complete")

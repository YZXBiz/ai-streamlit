#!/bin/bash

# nb2py.sh - Convert Jupyter notebooks to Python scripts
# Usage: ./nb2py.sh notebook.ipynb [output_filename.py]

set -e  # Exit immediately if a command exits with a non-zero status

if [ $# -lt 1 ]; then
    echo "Error: Missing notebook filename"
    echo "Usage: $0 notebook.ipynb [output_filename.py]"
    exit 1
fi

NOTEBOOK_FILE="$1"
OUTPUT_FILE="${2:-${NOTEBOOK_FILE%.ipynb}.py}"  # Use second param if provided, otherwise default to same name with .py

# Check if input file exists
if [ ! -f "$NOTEBOOK_FILE" ]; then
    echo "Error: Notebook file '$NOTEBOOK_FILE' not found"
    exit 1
fi

echo "Converting '$NOTEBOOK_FILE' to Python script..."

# Run the conversion with uv
uv run -m jupyter nbconvert --to script --no-prompt "$NOTEBOOK_FILE" --output "$(basename "$OUTPUT_FILE" .py)"

echo "Conversion complete: $(dirname "$NOTEBOOK_FILE")/$(basename "$OUTPUT_FILE")"
echo "✅ Done!"

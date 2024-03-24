#!/bin/bash

# Run the first Python script
python3 pack_to_npz.py

# Check if the first script executed successfully
if [ $? -eq 0 ]; then
    echo "First script executed successfully."
else
    echo "Error: First script failed to execute."
    exit 1
fi

# Run the second Python script
python3 hf_upload.py

# Check if the second script executed successfully
if [ $? -eq 0 ]; then
    echo "Second script executed successfully."
else
    echo "Error: Second script failed to execute."
    exit 1
fi

echo "Both scripts executed successfully."

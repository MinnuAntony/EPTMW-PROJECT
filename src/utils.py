# utils.py content

import json
import os
import joblib

def save_json(data, filename):
    """Save dictionary data to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved data to {filename}")

def load_json(filename):
    """Load dictionary data from a JSON file."""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    else:
        print(f"File {filename} not found.")
        return {}

def save_model(model, filename):
    """Save model to a file using joblib."""
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename):
    """Load model from a file using joblib."""
    if os.path.exists(filename):
        return joblib.load(filename)
    else:
        print(f"Model file {filename} not found.")
        return None

def print_banner(title):
    """Print a formatted banner for logging steps."""
    print("\n" + "=" * 60)
    print(f"{title.center(60)}")
    print("=" * 60 + "\n")

def check_file_exists(filepath):
    """Check if a file exists."""
    if os.path.exists(filepath):
        print(f"File found: {filepath}")
        return True
    else:
        print(f"File not found: {filepath}")
        return False

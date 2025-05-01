"""
Simple script to check if SNLI and LogiQA datasets exist and their sizes
"""

import os
from pathlib import Path

def check_file(file_path):
    """Check if a file exists and get its size."""
    if not os.path.exists(file_path):
        return False, 0
    
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    return True, size_mb

def main():
    """Main function to check dataset files."""
    # Check SNLI dataset
    snli_path = Path("training/datasets/nli/snli.json")
    snli_exists, snli_size = check_file(snli_path)
    
    print(f"SNLI dataset: {'Exists' if snli_exists else 'Not found'}")
    if snli_exists:
        print(f"  - Size: {snli_size:.2f} MB")
    
    # Check LogiQA dataset
    logiqa_path = Path("training/datasets/logiqa/logiqa.json")
    logiqa_exists, logiqa_size = check_file(logiqa_path)
    
    print(f"LogiQA dataset: {'Exists' if logiqa_exists else 'Not found'}")
    if logiqa_exists:
        print(f"  - Size: {logiqa_size:.2f} MB")
    
    # Overall status
    if snli_exists and logiqa_exists:
        print("\nBoth datasets are ready to use!")
    else:
        missing = []
        if not snli_exists:
            missing.append("SNLI")
        if not logiqa_exists:
            missing.append("LogiQA")
            
        print(f"\nMissing datasets: {', '.join(missing)}")

if __name__ == "__main__":
    main() 
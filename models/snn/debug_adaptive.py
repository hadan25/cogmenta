#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Debug script for adaptive spike processor
"""

import sys
import os
import traceback

def main():
    try:
        print("Trying to import AdaptiveSpikeProcessor...")
        from models.snn.adaptive_spike_processor import AdaptiveSpikeProcessor
        print("Import successful!")
        
        # Try to create an instance
        processor = AdaptiveSpikeProcessor()
        print(f"Created processor: {processor}")
        
    except ImportError as e:
        print(f"Import error: {e}")
        traceback.print_exc()
        
        # Try alternate import path
        try:
            print("\nTrying alternate import path...")
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
            from models.snn.adaptive_spike_processor import AdaptiveSpikeProcessor
            print("Alternate import successful!")
        except Exception as e2:
            print(f"Alternate import failed: {e2}")
            traceback.print_exc()
    
    except Exception as e:
        print(f"Other error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    print(f"sys.path: {sys.path}")
    main() 
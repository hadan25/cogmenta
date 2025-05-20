"""
Test script for AEDAT converter implementation.

This script tests the functionality of the AEDAT event data converter
by loading CIFAR10-DVS AEDAT files and converting them to spike representation.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import converters
from training.utils.data_to_spike import (
    AEDATEventToSpikeConverter, 
    AEDAT_AVAILABLE, 
    visualize_spike_data
)

# Path to CIFAR10-DVS dataset
CIFAR10_DVS_PATH = "training/datasets/cifar10_dvs"

def test_aedat_converter(file_path=None, class_name=None):
    """
    Test AEDAT converter with a CIFAR10-DVS file.
    
    Args:
        file_path: Path to AEDAT file (optional)
        class_name: Class name to use from CIFAR10-DVS (optional)
        
    Returns:
        Success status
    """
    if not AEDAT_AVAILABLE:
        print("AEDAT library not available. Please install with 'pip install aedat'")
        return False
    
    # Check for file path
    if file_path is None:
        # Use CIFAR10-DVS dataset
        cifar_path = Path(CIFAR10_DVS_PATH)
        
        if not cifar_path.exists():
            print(f"CIFAR10-DVS dataset not found at {cifar_path}")
            return False
        
        # Get available classes
        classes = [d.name for d in cifar_path.iterdir() if d.is_dir()]
        
        if not classes:
            print("No class directories found in CIFAR10-DVS dataset")
            return False
        
        # Select class
        if class_name is None or class_name not in classes:
            class_name = random.choice(classes)
            print(f"Randomly selected class: {class_name}")
        
        # Get AEDAT files for this class
        class_dir = cifar_path / class_name
        aedat_files = list(class_dir.glob("*.aedat"))
        
        if not aedat_files:
            print(f"No AEDAT files found in {class_dir}")
            return False
        
        # Select a random file
        file_path = str(random.choice(aedat_files))
        print(f"Selected AEDAT file: {file_path}")
    
    # Create converter
    neuron_count = 1000
    converter = AEDATEventToSpikeConverter(
        neuron_count=neuron_count,
        spatial_dimensions=(128, 128)  # CIFAR10-DVS uses 128x128 DVS
    )
    
    try:
        print(f"Converting AEDAT file: {file_path}")
        
        # Convert to spike representation
        spike_data = converter.convert(file_path)
        
        # Check results
        if 'times' in spike_data and 'units' in spike_data and len(spike_data['times']) > 0:
            print(f"Conversion successful!")
            print(f"Number of spikes: {len(spike_data['times'])}")
            print(f"Time range: {spike_data['times'].min():.2f} to {spike_data['times'].max():.2f} ms")
            print(f"Neuron indices range: {spike_data['units'].min()} to {spike_data['units'].max()}")
            print(f"Number of active neurons: {len(np.unique(spike_data['units']))}")
            
            # Visualize results if enough spikes
            if len(spike_data['times']) > 10:
                try:
                    # Create output directory if it doesn't exist
                    output_dir = Path("output")
                    output_dir.mkdir(exist_ok=True)
                    
                    # Visualize spike data
                    class_label = class_name if class_name else Path(file_path).parent.name
                    title = f"CIFAR10-DVS Spike Representation - {class_label}"
                    save_path = output_dir / f"cifar10_dvs_{class_label}_spikes.png"
                    
                    # Use our visualization function
                    visualize_spike_data(
                        spike_data,
                        title=title,
                        max_points=2000,
                        save_path=save_path
                    )
                    
                    print(f"Visualization saved to {save_path}")
                    
                    # Also create a time surface visualization (2D representation)
                    try:
                        # Create 2D histogram of events
                        if 'polarities' in spike_data:
                            # Use polarities to color the events
                            polarities = spike_data['polarities']
                            on_indices = polarities == 1
                            off_indices = polarities == 0
                            
                            plt.figure(figsize=(10, 8))
                            
                            # Plot ON events in red
                            if np.any(on_indices):
                                plt.scatter(
                                    spike_data['times'][on_indices][:1000], 
                                    spike_data['units'][on_indices][:1000],
                                    alpha=0.5, s=3, c='red', label='ON events'
                                )
                            
                            # Plot OFF events in blue
                            if np.any(off_indices):
                                plt.scatter(
                                    spike_data['times'][off_indices][:1000], 
                                    spike_data['units'][off_indices][:1000],
                                    alpha=0.5, s=3, c='blue', label='OFF events'
                                )
                                
                            plt.legend()
                            plt.title(f"ON/OFF Events - {class_label}")
                            plt.xlabel('Time (ms)')
                            plt.ylabel('Neuron Index')
                            
                            # Save polarity visualization
                            polarity_path = output_dir / f"cifar10_dvs_{class_label}_polarity.png"
                            plt.savefig(polarity_path)
                            plt.close()
                            print(f"Polarity visualization saved to {polarity_path}")
                        
                    except Exception as e:
                        print(f"Error creating time surface visualization: {e}")
                
                except Exception as e:
                    print(f"Error creating visualization: {e}")
            
            return True
        else:
            print("Conversion failed: Invalid spike data format or no spikes found")
            return False
    
    except Exception as e:
        print(f"Error testing AEDAT converter: {e}")
        return False

def test_all_classes():
    """
    Test AEDAT converter with one file from each class in CIFAR10-DVS.
    
    Returns:
        Number of successful conversions
    """
    cifar_path = Path(CIFAR10_DVS_PATH)
    if not cifar_path.exists():
        print(f"CIFAR10-DVS dataset not found at {cifar_path}")
        return 0
    
    # Get available classes
    classes = [d.name for d in cifar_path.iterdir() if d.is_dir()]
    
    if not classes:
        print("No class directories found in CIFAR10-DVS dataset")
        return 0
    
    print(f"Found {len(classes)} classes: {classes}")
    
    # Test each class
    success_count = 0
    for class_name in classes:
        print(f"\n\nTesting class: {class_name}")
        if test_aedat_converter(class_name=class_name):
            success_count += 1
    
    print(f"\nSuccessfully converted {success_count} out of {len(classes)} classes")
    return success_count

if __name__ == "__main__":
    # Get file path from command line arguments if provided
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            # Test all classes
            test_all_classes()
        else:
            # Test specific file
            file_path = sys.argv[1]
            test_aedat_converter(file_path)
    else:
        # Test a random file
        test_aedat_converter() 
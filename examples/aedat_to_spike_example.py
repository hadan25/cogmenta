"""
Example script demonstrating how to use the AEDAT converter.

This script shows how to:
1. Load AEDAT files from neuromorphic cameras
2. Convert them to spike representations
3. Feed them to a Spiking Neural Network
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from training.utils.data_to_spike import (
    AEDATEventToSpikeConverter, 
    UniversalDataToSpikeConverter,
    visualize_spike_data, 
    AEDAT_AVAILABLE
)
from training.utils.dataset_loader import EventBasedDataset
from models.snn.perceptual_snn import PerceptualSNN

def process_aedat_file(file_path, output_dir=None):
    """
    Process an AEDAT file and convert it to spike representation.
    
    Args:
        file_path: Path to AEDAT file
        output_dir: Directory to save outputs (optional)
        
    Returns:
        Spike data dictionary
    """
    if not AEDAT_AVAILABLE:
        print("AEDAT library not available. Please install with 'pip install aedat'")
        return None
    
    if output_dir:
        Path(output_dir).mkdir(exist_ok=True)
    
    print(f"Processing AEDAT file: {file_path}")
    
    # Create specialized converter for AEDAT files
    neuron_count = 1000
    converter = AEDATEventToSpikeConverter(
        neuron_count=neuron_count,
        spatial_dimensions=(346, 260)  # Default DVS/DAVIS resolution
    )
    
    # Convert to spike representation
    spike_data = converter.convert(file_path)
    
    # Print statistics
    if 'times' in spike_data and len(spike_data['times']) > 0:
        print(f"Conversion successful!")
        print(f"Number of spikes: {len(spike_data['times'])}")
        print(f"Time range: {spike_data['times'].min():.2f} to {spike_data['times'].max():.2f} ms")
        print(f"Neuron indices range: {spike_data['units'].min()} to {spike_data['units'].max()}")
        print(f"Number of active neurons: {len(np.unique(spike_data['units']))}")
        
        # Visualize if output directory provided
        if output_dir:
            save_path = Path(output_dir) / f"{Path(file_path).stem}_spikes.png"
            visualize_spike_data(
                spike_data,
                title=f"AEDAT Spike Representation - {Path(file_path).name}",
                max_points=2000,
                save_path=save_path
            )
            print(f"Visualization saved to {save_path}")
    else:
        print("Conversion failed or no events found")
    
    return spike_data

def process_aedat_with_snn(file_path, output_dir=None):
    """
    Process an AEDAT file with a Spiking Neural Network.
    
    Args:
        file_path: Path to AEDAT file
        output_dir: Directory to save outputs (optional)
        
    Returns:
        SNN processing results
    """
    # First convert to spike representation
    spike_data = process_aedat_file(file_path, output_dir)
    
    if spike_data is None or 'times' not in spike_data or len(spike_data['times']) == 0:
        print("Failed to convert AEDAT file to spikes")
        return None
    
    # Create a Perceptual SNN for event-based data
    neuron_count = 1000
    snn = PerceptualSNN(
        neuron_count=neuron_count,
        topology_type="flexible",
        modality="visual"  # Use visual modality for event data
    )
    
    print("\nProcessing with SNN...")
    
    # Prepare input format for SNN
    input_data = {
        'spikes': {
            'times': spike_data['times'],
            'units': spike_data['units'],
            'mask': np.ones(len(spike_data['times']), dtype=bool)
        }
    }
    
    # Process with SNN
    result = snn.process_input(input_data)
    
    # Print results
    print(f"SNN processing complete")
    print(f"Phi (integration): {result.get('phi', 'N/A')}")
    
    # Print active regions
    if 'region_activations' in result:
        print("\nRegion activations:")
        for region, activation in result['region_activations'].items():
            print(f"  {region}: {activation:.3f}")
    
    return result

def batch_process_aedat_files(data_dir, output_dir=None):
    """
    Process all AEDAT files in a directory.
    
    Args:
        data_dir: Directory containing AEDAT files
        output_dir: Directory to save outputs (optional)
        
    Returns:
        Number of files processed
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Data directory not found: {data_dir}")
        return 0
    
    # Find all AEDAT files
    aedat_files = list(data_path.glob("**/*.aedat*"))
    
    if not aedat_files:
        print(f"No AEDAT files found in {data_dir}")
        return 0
    
    print(f"Found {len(aedat_files)} AEDAT files")
    
    # Process each file
    for i, file_path in enumerate(aedat_files):
        print(f"\nProcessing file {i+1}/{len(aedat_files)}: {file_path}")
        process_aedat_file(str(file_path), output_dir)
    
    return len(aedat_files)

def demo_universal_converter(file_path):
    """
    Demonstrate the UniversalDataToSpikeConverter with AEDAT files.
    
    Args:
        file_path: Path to AEDAT file
        
    Returns:
        Spike data dictionary
    """
    if not AEDAT_AVAILABLE:
        print("AEDAT library not available. Please install with 'pip install aedat'")
        return None
    
    print(f"Testing UniversalDataToSpikeConverter with: {file_path}")
    
    # Create universal converter that automatically detects file type
    neuron_count = 1000
    universal_converter = UniversalDataToSpikeConverter(neuron_count)
    
    # Convert to spike representation
    spike_data = universal_converter.convert(file_path)
    
    # Print statistics
    if 'times' in spike_data and len(spike_data['times']) > 0:
        print(f"Universal converter successful!")
        print(f"Number of spikes: {len(spike_data['times'])}")
        print(f"Time range: {spike_data['times'].min():.2f} to {spike_data['times'].max():.2f} ms")
    else:
        print("Universal converter failed or no events found")
    
    return spike_data

def demo_event_dataset(data_dir):
    """
    Demonstrate using the EventBasedDataset class.
    
    Args:
        data_dir: Directory containing event-based data
        
    Returns:
        Dataset object
    """
    print(f"Creating EventBasedDataset from: {data_dir}")
    
    # Create dataset
    dataset = EventBasedDataset(data_dir)
    
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Access a sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample shape: {sample['sample'].shape}")
        print(f"Sample label: {sample['label']}")
    
    return dataset

if __name__ == "__main__":
    # Get file or directory path from command line arguments
    if len(sys.argv) > 1:
        path = sys.argv[1]
        output_dir = "output"
        
        if os.path.isfile(path):
            # Process single file
            if path.lower().endswith(('.aedat', '.aedat4')):
                # Process with SNN
                process_aedat_with_snn(path, output_dir)
                
                # Demo universal converter
                demo_universal_converter(path)
            else:
                print(f"Not an AEDAT file: {path}")
        
        elif os.path.isdir(path):
            # Process directory
            batch_process_aedat_files(path, output_dir)
            
            # Demo dataset
            demo_event_dataset(path)
        
        else:
            print(f"Path not found: {path}")
    else:
        print("Usage: python aedat_to_spike_example.py <path_to_aedat_file_or_directory>")
        
        # Look for any AEDAT files in the data directory
        data_dir = Path("data")
        if data_dir.exists():
            aedat_files = list(data_dir.glob("**/*.aedat*"))
            if aedat_files:
                print(f"\nFound AEDAT file: {aedat_files[0]}")
                print(f"Try running: python {sys.argv[0]} {aedat_files[0]}")
            else:
                print("\nNo AEDAT files found in data directory.")
        else:
            print("\nData directory not found.") 
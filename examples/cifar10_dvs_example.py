"""
Example script demonstrating how to use the CIFAR10-DVS dataset with the AEDAT converter.

This script shows how to:
1. Load CIFAR10-DVS dataset (neuromorphic version of CIFAR-10)
2. Convert AEDAT files to spike representations
3. Train a simple SNN classifier on the dataset
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import random
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from training.utils.data_to_spike import (
    AEDATEventToSpikeConverter, 
    visualize_spike_data, 
    AEDAT_AVAILABLE
)
from training.utils.dataset_loader import EventBasedDataset
from torch.utils.data import DataLoader, random_split

# Path to CIFAR10-DVS dataset
CIFAR10_DVS_PATH = "training/datasets/cifar10_dvs"

def visualize_cifar10_dvs_samples(num_samples=5):
    """
    Visualize random samples from the CIFAR10-DVS dataset.
    
    Args:
        num_samples: Number of samples to visualize per class
    """
    if not AEDAT_AVAILABLE:
        print("AEDAT library not available. Please install with 'pip install aedat'")
        return
    
    cifar_path = Path(CIFAR10_DVS_PATH)
    if not cifar_path.exists():
        print(f"CIFAR10-DVS dataset not found at {cifar_path}")
        return
    
    # Get available classes
    classes = [d.name for d in cifar_path.iterdir() if d.is_dir()]
    
    if not classes:
        print("No class directories found in CIFAR10-DVS dataset")
        return
    
    print(f"Found {len(classes)} classes: {', '.join(classes)}")
    
    # Create output directory
    output_dir = Path("output/cifar10_dvs_samples")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create converter
    neuron_count = 1000
    converter = AEDATEventToSpikeConverter(
        neuron_count=neuron_count,
        spatial_dimensions=(128, 128)  # CIFAR10-DVS uses 128x128 DVS
    )
    
    # Process each class
    for class_name in classes:
        print(f"\nProcessing class: {class_name}")
        
        # Get AEDAT files for this class
        class_dir = cifar_path / class_name
        aedat_files = list(class_dir.glob("*.aedat"))
        
        if not aedat_files:
            print(f"No AEDAT files found in {class_dir}")
            continue
        
        # Select random samples
        selected_files = random.sample(aedat_files, min(num_samples, len(aedat_files)))
        
        # Process each file
        for i, file_path in enumerate(selected_files):
            try:
                print(f"  Converting {file_path.name}")
                
                # Convert to spike representation
                spike_data = converter.convert(str(file_path))
                
                if 'times' in spike_data and len(spike_data['times']) > 0:
                    # Visualize spike data
                    save_path = output_dir / f"{class_name}_{i}_spikes.png"
                    visualize_spike_data(
                        spike_data,
                        title=f"{class_name} - Sample {i+1}",
                        max_points=2000,
                        save_path=save_path
                    )
                    
                    # Create polarity visualization if available
                    if 'polarities' in spike_data:
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
                        plt.title(f"{class_name} - Sample {i+1} (ON/OFF Events)")
                        plt.xlabel('Time (ms)')
                        plt.ylabel('Neuron Index')
                        
                        # Save polarity visualization
                        polarity_path = output_dir / f"{class_name}_{i}_polarity.png"
                        plt.savefig(polarity_path)
                        plt.close()
                
            except Exception as e:
                print(f"  Error processing {file_path.name}: {e}")

def load_cifar10_dvs_dataset(batch_size=32, max_samples_per_class=100, val_split=0.2):
    """
    Load CIFAR10-DVS dataset using the EventBasedDataset class.
    
    Args:
        batch_size: Batch size for DataLoader
        max_samples_per_class: Maximum number of samples to load per class (for faster processing)
        val_split: Validation split ratio
        
    Returns:
        train_loader, val_loader, class_names
    """
    # Create dataset
    dataset = EventBasedDataset(CIFAR10_DVS_PATH, max_events=5000)
    
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Get class names
    cifar_path = Path(CIFAR10_DVS_PATH)
    class_names = [d.name for d in cifar_path.iterdir() if d.is_dir()]
    
    # Split into train and validation sets
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, class_names

def extract_features_from_cifar10_dvs():
    """
    Extract features from CIFAR10-DVS dataset and save them.
    
    This pre-processes the AEDAT files and extracts useful features for classification.
    """
    if not AEDAT_AVAILABLE:
        print("AEDAT library not available. Please install with 'pip install aedat'")
        return
    
    cifar_path = Path(CIFAR10_DVS_PATH)
    if not cifar_path.exists():
        print(f"CIFAR10-DVS dataset not found at {cifar_path}")
        return
    
    # Get available classes
    classes = [d.name for d in cifar_path.iterdir() if d.is_dir()]
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    print(f"Found {len(classes)} classes: {', '.join(classes)}")
    
    # Create output directory
    output_dir = Path("output/cifar10_dvs_features")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create converter
    neuron_count = 1000
    converter = AEDATEventToSpikeConverter(
        neuron_count=neuron_count,
        spatial_dimensions=(128, 128)  # CIFAR10-DVS uses 128x128 DVS
    )
    
    # Store features and labels
    all_features = []
    all_labels = []
    
    # Process each class
    for class_name in classes:
        print(f"\nProcessing class: {class_name}")
        
        # Get AEDAT files for this class
        class_dir = cifar_path / class_name
        aedat_files = list(class_dir.glob("*.aedat"))
        
        if not aedat_files:
            print(f"No AEDAT files found in {class_dir}")
            continue
        
        print(f"Found {len(aedat_files)} files, processing {min(20, len(aedat_files))} samples")
        
        # Process files with progress bar
        successful = 0
        for file_path in tqdm(aedat_files[:20], desc=f"Class {class_name}"):  # Limit to 20 files per class for faster processing
            try:
                # Use raw event data instead of spike representation
                events = converter._parse_aedat3_binary(str(file_path))
                
                if len(events) > 0:
                    # Extract features directly from raw events
                    features = extract_event_features(events)
                    
                    # Store features and label
                    all_features.append(features)
                    all_labels.append(class_to_idx[class_name])
                    successful += 1
                else:
                    print(f"  Warning: No events found in {file_path.name}")
            
            except Exception as e:
                print(f"  Error processing {file_path.name}: {e}")
        
        print(f"Successfully processed {successful} files from class {class_name}")
    
    # Convert to numpy arrays
    X = np.array(all_features)
    y = np.array(all_labels)
    
    # Save features and labels
    np.save(output_dir / "features.npy", X)
    np.save(output_dir / "labels.npy", y)
    
    print(f"\nExtracted features from {len(all_features)} samples")
    print(f"Feature shape: {X.shape}")
    print(f"Saved to {output_dir}")
    
    return X, y

def extract_event_features(events):
    """
    Extract features directly from event data.
    
    Args:
        events: Array of events with shape (n_events, 4) for [x, y, t, p]
        
    Returns:
        Feature vector
    """
    # Skip empty events
    if len(events) == 0:
        return np.zeros(20)
    
    # Extract x, y, t, p columns
    x = events[:, 0].astype(np.int16)
    y = events[:, 1].astype(np.int16)
    t = events[:, 2].astype(np.float32)
    p = events[:, 3].astype(np.int8)
    
    # Normalize timestamps
    if np.max(t) > np.min(t):
        t = (t - np.min(t)) / (np.max(t) - np.min(t)) * 1000  # Scale to [0, 1000] ms
    
    # Basic features
    features = [
        len(events),  # Number of events
        len(np.unique(x)),  # Number of unique x coordinates
        len(np.unique(y)),  # Number of unique y coordinates
        np.mean(t),  # Mean event time
        np.std(t),  # Std of event times
    ]
    
    # Count events in time windows
    time_min, time_max = np.min(t), np.max(t)
    time_span = max(1, time_max - time_min)
    window_size = time_span / 5
    
    for i in range(5):
        window_start = time_min + i * window_size
        window_end = window_start + window_size
        window_events = np.sum((t >= window_start) & (t < window_end))
        features.append(window_events / max(1, len(t)))  # Normalized event count
    
    # Polarity features
    on_ratio = np.sum(p == 1) / max(1, len(p))
    features.append(on_ratio)
    features.append(1 - on_ratio)  # OFF ratio
    
    # Spatial distribution features
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    features.append(x_mean / 128)  # Normalized x center
    features.append(y_mean / 128)  # Normalized y center
    
    # Spatial spread
    x_std = np.std(x) / 128
    y_std = np.std(y) / 128
    features.append(x_std)
    features.append(y_std)
    
    # Event density in quadrants
    for quad_x in range(2):
        for quad_y in range(2):
            x_mask = (x >= quad_x * 64) & (x < (quad_x + 1) * 64)
            y_mask = (y >= quad_y * 64) & (y < (quad_y + 1) * 64)
            quad_events = np.sum(x_mask & y_mask)
            features.append(quad_events / max(1, len(events)))
    
    return np.array(features)

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "visualize":
            # Visualize samples
            num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            visualize_cifar10_dvs_samples(num_samples)
        
        elif sys.argv[1] == "extract":
            # Extract features
            extract_features_from_cifar10_dvs()
        
        elif sys.argv[1] == "dataset":
            # Test dataset loading
            train_loader, val_loader, class_names = load_cifar10_dvs_dataset()
            print(f"Classes: {class_names}")
            print(f"Train batches: {len(train_loader)}")
            print(f"Val batches: {len(val_loader)}")
            
            # Show a batch
            for batch in train_loader:
                samples = batch['sample']
                labels = batch['label']
                print(f"Batch shape: {samples.shape}")
                print(f"Labels: {labels}")
                break
    
    else:
        print("Usage:")
        print("  python cifar10_dvs_example.py visualize [num_samples]")
        print("  python cifar10_dvs_example.py extract")
        print("  python cifar10_dvs_example.py dataset") 
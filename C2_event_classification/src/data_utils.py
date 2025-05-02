import uproot
import awkward as ak
import numpy as np
import pandas as pd
import os
import h5py
from datetime import datetime
import tensorflow as tf
def load_root_file(filepath, tree_name="events"):
    """
    Load a ROOT file and return the event tree as an Awkward Array.
    
    Parameters:
        filepath (str): Path to the ROOT file.
        tree_name (str): Name of the tree inside the ROOT file (default: "events").
    
    Returns:
        ak.Array: Awkward array of event data.
    """
    tree = uproot.open(f"{filepath}:{tree_name}")
    return tree.arrays(library="ak")

def save_datasets_hdf5(datasets, labels, masks, output_dir="../datasets"):
    """
    Save preprocessed datasets to HDF5 format
    
    Args:
        datasets: Dict containing 'train', 'val', 'test' dictionaries with 'global', 'particle', 'sv' arrays
        labels: Dict containing 'train', 'val', 'test' label arrays
        masks: Dict containing particle and sv masks for each split
        output_dir: Directory to save the HDF5 file
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_path = os.path.join(output_dir, f"processed_data_{timestamp}.h5")
    
    with h5py.File(file_path, 'w') as f:
        # Store datasets
        for split in ['train', 'val', 'test']:
            # Create group for this split
            split_group = f.create_group(split)
            
            # Store features
            features_group = split_group.create_group('features')
            features_group.create_dataset('global', data=datasets[split]['global'])
            features_group.create_dataset('particle', data=datasets[split]['particle'])
            features_group.create_dataset('sv', data=datasets[split]['sv'])
            
            # Store masks
            masks_group = split_group.create_group('masks')
            masks_group.create_dataset('particle_mask', data=masks[split]['particle_mask'])
            masks_group.create_dataset('sv_mask', data=masks[split]['sv_mask'])
            
            # Store labels
            split_group.create_dataset('labels', data=labels[split])
    
    print(f"Datasets saved to {file_path}")
    return file_path

def load_datasets_hdf5(file_path, batch_size=64, return_numpy=True):
    """
    Load datasets from HDF5 file
    
    Args:
        file_path: Path to HDF5 file
        batch_size: Batch size for TF datasets
        return_numpy: If True, returns numpy arrays; if False, returns TF datasets
        
    Returns:
        If return_numpy=True: Dict with numpy arrays
        If return_numpy=False: Dict with TF datasets
    """
    numpy_data = {}
    tf_datasets = {}
    
    with h5py.File(file_path, 'r') as f:
        for split in ['train', 'val', 'test']:
            # Load data
            global_array = f[f'{split}/features/global'][:]
            particle_array = f[f'{split}/features/particle'][:]
            sv_array = f[f'{split}/features/sv'][:]
            particle_mask = f[f'{split}/masks/particle_mask'][:]
            sv_mask = f[f'{split}/masks/sv_mask'][:]
            labels = f[f'{split}/labels'][:]
            
            # Store numpy arrays
            numpy_data[split] = {
                'global': global_array,
                'particle': particle_array,
                'sv': sv_array,
                'particle_mask': particle_mask,
                'sv_mask': sv_mask,
                'labels': labels
            }
            
            # Create TF Dataset
            tf_datasets[split] = create_tf_dataset(
                global_array, particle_array, particle_mask, 
                sv_array, sv_mask, labels, batch_size
            )
    
    return numpy_data if return_numpy else tf_datasets

def create_tf_dataset(global_array, particle_array, particle_mask, sv_array, sv_mask, labels, batch_size=64):
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            "global_input": global_array,
            "particle_input": particle_array,
            "particle_mask": particle_mask,
            "sv_input": sv_array,
            "sv_mask": sv_mask
        },
        labels
    ))
    return dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
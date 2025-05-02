import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import logging
from src.data_utils import load_datasets_hdf5
from src.model_utils.model import build_classifier

def train_model(data_file, output_dir="./models", epochs=50, batch_size=64, patience=10):
    """
    Train the event classifier model with early stopping and learning rate scheduling.
    
    Parameters:
        data_file (str): Path to the HDF5 file containing preprocessed datasets
        output_dir (str): Directory to save model and training outputs
        epochs (int): Maximum number of epochs to train
        batch_size (int): Batch size for training
        patience (int): Number of epochs to wait before early stopping
        
    Returns:
        dict: Training history
        str: Path to saved model
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Load datasets
    print(f"Loading datasets from {data_file}")
    tf_datasets = load_datasets_hdf5(data_file, batch_size=batch_size, return_numpy=False)
    train_ds = tf_datasets['train']
    val_ds = tf_datasets['val']
    
    # Build model
    print("Building model...")
    model = build_classifier()
    
    # Setup callbacks
    callbacks = [
        # Early stopping to prevent overfitting
        tf.keras.callbacks.EarlyStopping(
            monitor='val_sparse_categorical_accuracy',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        # Learning rate scheduler
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1
        ),
        # Model checkpointing - save the best model based on validation accuracy
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(run_dir, 'checkpoints/model_{epoch:02d}_{val_sparse_categorical_accuracy:.4f}.keras'),
            monitor='val_sparse_categorical_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(run_dir, 'logs'),
            histogram_freq=1
        )
    ]
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top_2_accuracy')
        ]
    )
    
    # Print model summary
    model.summary()
    
    # Train model
    print(f"Starting training for {epochs} epochs...")
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model_path = os.path.join(run_dir, 'final_model.keras')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Save training history
    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'sparse_categorical_accuracy': [float(x) for x in history.history['sparse_categorical_accuracy']],
        'val_sparse_categorical_accuracy': [float(x) for x in history.history['val_sparse_categorical_accuracy']],
        'top_2_accuracy': [float(x) for x in history.history['top_2_accuracy']],
        'val_top_2_accuracy': [float(x) for x in history.history['val_top_2_accuracy']]
    }
    
    with open(os.path.join(run_dir, 'training_history.json'), 'w') as f:
        json.dump(history_dict, f, indent=4)
    
    # Plot and save training curves
    plot_training_curves(history, run_dir)
    
    return history, model_path


def plot_training_curves(history, output_dir):
    """
    Plot and save training and validation metrics curves.
    
    Parameters:
        history: Training history object
        output_dir: Directory to save plots
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curve
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curve
    ax2.plot(history.history['sparse_categorical_accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_sparse_categorical_accuracy'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.pdf'), dpi=300)
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train the event classifier model")
    parser.add_argument("--data_file", type=str, required=True, help="Path to HDF5 data file")
    parser.add_argument("--output_dir", type=str, default="./models", help="Output directory for model and logs")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    
    args = parser.parse_args()
    
    train_model(
        data_file=args.data_file,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience
    ) 
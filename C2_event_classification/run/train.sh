#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=/home/xl628/courseworks/a3-cw/logs/train.log
#SBATCH --error=/home/xl628/courseworks/a3-cw/logs/train.err
#SBATCH --time=12:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:4
#SBATCH -A MPHIL-DIS-SL2-GPU

# CSD3 submit: sbatch scripts/train.sh
# CSD3 list submitted jobs: gstatement
# squeue -p ampere

# Please run this script at the ROOT of the repository ./run/train.sh

# cd $SLURM_SUBMIT_DIR
# module load cuda/11.4
# source .venv/bin/activate

# Default values
DATA_FILE="datasets/processed_data_20250411-011241.h5"
OUTPUT_DIR="run/models"
EPOCHS=50
BATCH_SIZE=256
PATIENCE=10 # early stopping patience

# Check if data_file is provided
if [ -z "$DATA_FILE" ]; then
  echo "Error: --data_file is required"
  echo "Usage: ./train.sh --data_file <path_to_hdf5> [--output_dir <dir>] [--epochs <num>] [--batch_size <size>] [--patience <num>]"
  exit 1
fi

# Run the training script
PYTHONPATH=. python run/train.py \
  --data_file "$DATA_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --patience "$PATIENCE"
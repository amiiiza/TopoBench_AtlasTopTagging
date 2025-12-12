#!/bin/bash
#SBATCH --job-name=atlas-gat
#SBATCH --time=05:00:00
#SBATCH --gpus=v100:1
#SBATCH --partition=gpu-v100-32g
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --output=logs/atlas_gat_%j.out
#SBATCH --error=logs/atlas_gat_%j.err

# Configuration
SUBSET=0.05
EXPERIMENT="atlas_top_tagging/gat"
WANDB_PROJECT="ATLAS_TopTagging"
WANDB_RUN_NAME="gat_subset${SUBSET}_${SLURM_JOB_ID}" 

echo "=========================================="
echo "ATLAS Training Started"
echo "=========================================="
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Experiment: $EXPERIMENT"
echo "Subset: ${SUBSET}"
echo "WandB Project: $WANDB_PROJECT"
echo "WandB Run: $WANDB_RUN_NAME"
echo "=========================================="
echo ""

# Setup
mkdir -p logs
module load mamba
source activate /scratch/work/akbaria1/.conda_envs/tbAtlas
cd /scratch/work/akbaria1/atlas/Challenge_TopoBench/TopoBench

# WandB Configuration
export WANDB_PROJECT="ATLAS_TopTagging"
export WANDB_NAME="${EXPERIMENT}_subset${SUBSET}_job${SLURM_JOB_ID}"
export WANDB_MODE=online

# Run training
START_TIME=$(date +%s)

python -m topobench \
    experiment=$EXPERIMENT \
    dataset.loader.parameters.subset=$SUBSET \
    trainer.max_epochs=100

EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

echo ""
echo "=========================================="
echo "ATLAS Training Finished"
echo "=========================================="
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Duration: ${HOURS}h ${MINUTES}m"
echo "Exit code: $EXIT_CODE"
echo "View results at: https://wandb.ai/YOUR_USERNAME/$WANDB_PROJECT"
echo "=========================================="
#!/bin/bash
#SBATCH --job-name=atlas-all
#SBATCH --time=6:00:00
#SBATCH --gpus=v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --output=logs/atlas_all_%j.out
#SBATCH --error=logs/atlas_all_%j.err

echo "=========================================="
echo "ATLAS All Experiments Test"
echo "=========================================="
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Setup
mkdir -p logs
module load mamba
source activate /scratch/work/akbaria1/.conda_envs/tbAtlas
cd /scratch/work/akbaria1/atlas/Challenge_TopoBench/TopoBench

# WandB Configuration
export WANDB_PROJECT="ATLAS_TopTagging"
export WANDB_MODE=online

SUBSET=0.001  # Very small for quick testing
EPOCHS=5      # Few epochs for quick testing

# List of experiments to test
EXPERIMENTS=(
    "atlas_top_tagging/mlp"
    "atlas_top_tagging/detector_layer"
    "atlas_top_tagging/detector_layer_no_maxdiff"
    "atlas_top_tagging/same_particle"
    "atlas_top_tagging/same_particle_no_maxdiff"
)

# Run each experiment
for EXP in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running: $EXP"
    echo "=========================================="
    
    export WANDB_NAME="${EXP}_subset${SUBSET}_job${SLURM_JOB_ID}"
    
    python -m topobench \
        experiment=$EXP \
        dataset.loader.parameters.subset=$SUBSET \
        trainer.max_epochs=$EPOCHS
    
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ $EXP completed successfully"
    else
        echo "❌ $EXP failed with exit code $EXIT_CODE"
    fi
done

echo ""
echo "=========================================="
echo "All Experiments Completed"
echo "=========================================="
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
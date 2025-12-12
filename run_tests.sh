#!/bin/bash
#SBATCH --job-name=atlas-tests
#SBATCH --time=00:30:00
#SBATCH --gpus=v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err

echo "=========================================="
echo "ATLAS TopoBench Tests"
echo "=========================================="
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="
echo ""

# Setup
mkdir -p logs
module load mamba
source activate /scratch/work/akbaria1/.conda_envs/tbAtlas
cd /scratch/work/akbaria1/atlas/Challenge_TopoBench/TopoBench

# Install pytest if needed
pip install pytest pytest-cov --quiet --break-system-packages

echo ""
echo "Testing DetectorLayerLifting..."
echo "----------------------------------------"
pytest test/transforms/liftings/pointcloud2hypergraph/test_detector_layer.py -v

echo ""
echo "Testing MaxDifference..."
echo "----------------------------------------"
pytest test/transforms/feature_liftings/test_MaxDifference.py -v

echo ""
echo "Running both tests with coverage..."
echo "----------------------------------------"
pytest test/transforms/liftings/pointcloud2hypergraph/test_detector_layer.py \
       test/transforms/feature_liftings/test_MaxDifference.py \
       --cov=topobench.transforms.liftings.pointcloud2hypergraph.detector_layer \
       --cov=topobench.transforms.feature_liftings.max_difference \
       --cov-report=term-missing \
       -v

echo ""
echo "=========================================="
echo "Tests Completed!"
echo "=========================================="
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Check logs/test_${SLURM_JOB_ID}.out for results"
echo "=========================================="
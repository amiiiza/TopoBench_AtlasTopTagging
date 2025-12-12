#!/bin/bash
#SBATCH --job-name=atlas-tests-coverage
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --output=logs/test_cover_%j.out
#SBATCH --error=logs/test_cover_%j.err

echo "=========================================="
echo "Coverage Check for ATLAS Dataset"
echo "=========================================="
echo "Start time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Load conda
mkdir -p logs
module load mamba
source activate /scratch/work/akbaria1/.conda_envs/tbAtlas
cd /scratch/work/akbaria1/atlas/Challenge_TopoBench/TopoBench

echo ""
echo "Running tests with coverage..."
echo ""

# Run pytest with coverage
pytest test/data/datasets/test_atlas_top_tagging_dataset.py \
    -v \
    --cov=topobench/data/datasets/atlas_top_tagging_dataset.py \
    --cov-report=term-missing \
    --cov-report=html:htmlcov \
    --cov-report=term \
    2>&1 | tee coverage_output.txt

echo ""
echo "=========================================="
echo "Coverage Summary"
echo "=========================================="

# Extract coverage percentage
grep "TOTAL" coverage_output.txt || echo "Could not find coverage total"

echo ""
echo "=========================================="
echo "Missing Lines (if any)"
echo "=========================================="

# Show missing lines
grep "atlas_top_tagging_dataset.py" coverage_output.txt | grep -v "100%"

echo ""
echo "=========================================="
echo "HTML Report Generated"
echo "=========================================="
echo "Location: $(pwd)/htmlcov/index.html"
echo ""
echo "To view:"
echo "1. Download htmlcov folder to your local machine"
echo "2. Open htmlcov/index.html in browser"
echo ""
echo "Or check detailed missing lines above"
echo "=========================================="

echo ""
echo "End time: $(date)"
echo "=========================================="

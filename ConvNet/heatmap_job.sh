#!/bin/bash -l

#SBATCH --job-name=heatmap_train
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --cpus-per-task=4
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python
conda activate traco

# Entpacken ins TMPDIR
tar -xf "$WORK/training.tar" -C "$TMPDIR"

# Optional: Debug-Ausgabe
echo "TMPDIR is $TMPDIR"
ls "$TMPDIR"

# Pfad setzen, damit 'traco' gefunden wird
export PYTHONPATH="$HOME"

# Ausführen
cd "$HOME/traco/ConvNet"
python heatmap_tracker_train.py
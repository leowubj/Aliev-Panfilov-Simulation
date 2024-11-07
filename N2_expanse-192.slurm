#!/bin/bash
#SBATCH --job-name="./apf"
#SBATCH --output="N2-192-8000-ALL.out"
#SBATCH --partition=compute
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=96
#SBATCH --mem=64G
#SBATCH --account=csd720
# #SBATCH --export=None
#SBATCH --export=ALL
#SBATCH -t 0:15:00
####   #SBATCH --mail-type=BEGIN,END,FAIL
####   #SBATCH --mail-user=your_email@ucsd.edu

# setup your environment

export SLURM_EXPORT_ENV=ALL
module purge
module load cpu
#Load module file(s) into the shell environment
module load gcc/9.2.0
module load mvapich2/2.3.6
module load slurm
srun --mpi=pmi2 -n 192 ./apf -n 8000 -i 8000 -x 1 -y 192
srun --mpi=pmi2 -n 192 ./apf -n 8000 -i 8000 -x 1 -y 192 -k
srun --mpi=pmi2 -n 192 ./apf -n 8000 -i 8000 -x 2 -y 96
srun --mpi=pmi2 -n 192 ./apf -n 8000 -i 8000 -x 2 -y 96 -k
srun --mpi=pmi2 -n 192 ./apf -n 8000 -i 8000 -x 3 -y 64
srun --mpi=pmi2 -n 192 ./apf -n 8000 -i 8000 -x 3 -y 64 -k
srun --mpi=pmi2 -n 192 ./apf -n 8000 -i 8000 -x 4 -y 48
srun --mpi=pmi2 -n 192 ./apf -n 8000 -i 8000 -x 4 -y 48 -k
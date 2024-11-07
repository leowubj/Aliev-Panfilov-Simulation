#!/bin/bash
#SBATCH --job-name="./apf"
#SBATCH --output="N1-192-10w-ALL.out"
#SBATCH --partition=compute
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=96
#SBATCH --mem=64G
#SBATCH --account=csd720
# #SBATCH --export=None
#SBATCH --export=ALL
#SBATCH -t 0:10:00
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
srun --mpi=pmi2 -n 192 ./apf -n 1800 -i 100000 -x 6 -y 32

srun --mpi=pmi2 -n 192 ./apf -n 1800 -i 100000 -x 8 -y 24

srun --mpi=pmi2 -n 192 ./apf -n 1800 -i 100000 -x 12 -y 16

srun --mpi=pmi2 -n 192 ./apf -n 1800 -i 100000 -x 16 -y 12

srun --mpi=pmi2 -n 192 ./apf -n 1800 -i 100000 -x 24 -y 8

srun --mpi=pmi2 -n 192 ./apf -n 1800 -i 100000 -x 32 -y 6

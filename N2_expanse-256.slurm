#!/bin/bash
#SBATCH --job-name="./apf"
#SBATCH --output="N2-256-8000-ALL.out"
#SBATCH --partition=compute
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=128
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
srun --mpi=pmi2 -n 256 ./apf -n 8000 -i 8000 -x 1 -y 256
srun --mpi=pmi2 -n 256 ./apf -n 8000 -i 8000 -x 1 -y 256 -k
srun --mpi=pmi2 -n 256 ./apf -n 8000 -i 8000 -x 2 -y 128
srun --mpi=pmi2 -n 256 ./apf -n 8000 -i 8000 -x 2 -y 128 -k
srun --mpi=pmi2 -n 256 ./apf -n 8000 -i 8000 -x 4 -y 64
srun --mpi=pmi2 -n 256 ./apf -n 8000 -i 8000 -x 4 -y 64 -k
srun --mpi=pmi2 -n 256 ./apf -n 8000 -i 8000 -x 8 -y 32
srun --mpi=pmi2 -n 256 ./apf -n 8000 -i 8000 -x 8 -y 32 -k
srun --mpi=pmi2 -n 256 ./apf -n 8000 -i 8000 -x 16 -y 16
srun --mpi=pmi2 -n 256 ./apf -n 8000 -i 8000 -x 16 -y 16 -k
srun --mpi=pmi2 -n 256 ./apf -n 8000 -i 8000 -x 32 -y 8
srun --mpi=pmi2 -n 256 ./apf -n 8000 -i 8000 -x 32 -y 8 -k
srun --mpi=pmi2 -n 256 ./apf -n 8000 -i 8000 -x 64 -y 4
srun --mpi=pmi2 -n 256 ./apf -n 8000 -i 8000 -x 64 -y 4 -k
srun --mpi=pmi2 -n 256 ./apf -n 8000 -i 8000 -x 128 -y 2
srun --mpi=pmi2 -n 256 ./apf -n 8000 -i 8000 -x 128 -y 2 -k
srun --mpi=pmi2 -n 256 ./apf -n 8000 -i 8000 -x 256 -y 1
srun --mpi=pmi2 -n 256 ./apf -n 8000 -i 8000 -x 256 -y 1 -k
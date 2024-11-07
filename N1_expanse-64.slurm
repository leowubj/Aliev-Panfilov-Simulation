#!/bin/bash
#SBATCH --job-name="./apf"
#SBATCH --output="N1-64-10w-ALL.out"
#SBATCH --partition=compute
#SBATCH --nodes=1
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
srun --mpi=pmi2 -n 64 ./apf -n 1800 -i 100000 -x 1 -y 64
srun --mpi=pmi2 -n 64 ./apf -n 1800 -i 100000 -x 1 -y 64 -k
srun --mpi=pmi2 -n 64 ./apf -n 1800 -i 100000 -x 2 -y 32
srun --mpi=pmi2 -n 64 ./apf -n 1800 -i 100000 -x 2 -y 32 -k
srun --mpi=pmi2 -n 64 ./apf -n 1800 -i 100000 -x 4 -y 16
srun --mpi=pmi2 -n 64 ./apf -n 1800 -i 100000 -x 4 -y 16 -k
srun --mpi=pmi2 -n 64 ./apf -n 1800 -i 100000 -x 8 -y 8
srun --mpi=pmi2 -n 64 ./apf -n 1800 -i 100000 -x 8 -y 8 -k
srun --mpi=pmi2 -n 64 ./apf -n 1800 -i 100000 -x 16 -y 4
srun --mpi=pmi2 -n 64 ./apf -n 1800 -i 100000 -x 16 -y 4 -k
srun --mpi=pmi2 -n 64 ./apf -n 1800 -i 100000 -x 32 -y 2
srun --mpi=pmi2 -n 64 ./apf -n 1800 -i 100000 -x 32 -y 2 -k
srun --mpi=pmi2 -n 64 ./apf -n 1800 -i 100000 -x 64 -y 1
srun --mpi=pmi2 -n 64 ./apf -n 1800 -i 100000 -x 64 -y 1 -k

#!/bin/bash
#SBATCH --job-name="./apf"
#SBATCH --output="apf.16.2000.100000.%j.%N.out"
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=64G
#SBATCH --account=csd720
#SBATCH --export=None
#SBATCH --export=ALL
#SBATCH -t 0:10:00
###   #SBATCH --mail-type=BEGIN,END,FAIL
###   #SBATCH --mail-user=your_email@ucsd.edu

# setup your environment

export SLURM_EXPORT_ENV=ALL
module purge
# module load cpu
module load cpu/0.15.4
#Load module file(s) into the shell environment
module load gcc/9.2.0
module load mvapich2/2.3.6
module load slurm
srun --mpi=pmi2 -n 1 ./apf -n 800 -i 2000 -x 1 -y 1 -k
srun --mpi=pmi2 -n 2 ./apf -n 800 -i 2000 -x 1 -y 2 -k
srun --mpi=pmi2 -n 3 ./apf -n 800 -i 2000 -x 1 -y 3 -k
srun --mpi=pmi2 -n 4 ./apf -n 800 -i 2000 -x 1 -y 4 -k
srun --mpi=pmi2 -n 5 ./apf -n 800 -i 2000 -x 1 -y 5 -k
srun --mpi=pmi2 -n 6 ./apf -n 800 -i 2000 -x 1 -y 6 -k
srun --mpi=pmi2 -n 7 ./apf -n 800 -i 2000 -x 1 -y 7 -k
srun --mpi=pmi2 -n 8 ./apf -n 800 -i 2000 -x 1 -y 8 -k
srun --mpi=pmi2 -n 9 ./apf -n 800 -i 2000 -x 1 -y 9 -k
srun --mpi=pmi2 -n 10 ./apf -n 800 -i 2000 -x 1 -y 10 -k
srun --mpi=pmi2 -n 11 ./apf -n 800 -i 2000 -x 1 -y 11 -k
srun --mpi=pmi2 -n 12 ./apf -n 800 -i 2000 -x 1 -y 12 -k
srun --mpi=pmi2 -n 13 ./apf -n 800 -i 2000 -x 1 -y 13 -k
srun --mpi=pmi2 -n 14 ./apf -n 800 -i 2000 -x 1 -y 14 -k
srun --mpi=pmi2 -n 15 ./apf -n 800 -i 2000 -x 1 -y 15 -k
srun --mpi=pmi2 -n 16 ./apf -n 800 -i 2000 -x 1 -y 16 -k
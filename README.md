# HPC - High Performance Computing
## Slurm (originally Simple Linux Utility for Resource Management)
### Log In: ```ssh awu53@euler.wacc.wisc.edu```
### Slurm used on Euler to share Euler’s hardware and software resources
* Uses a batch script as input -> **slurm is the name; ```sbatch``` is the utility**
* Partitions: Each partition has its own priority rules, job queue, hardware, and time limit (Euler uses partitions to enforce user priority)
* Threads: a thread describe a concurrent logical path of execution that can be run by a CPU (usually allocate one hardware thread per CPU, not assign threads from different jobs to the same physical CPU core)
```
#!/usr/bin/env zsh                     # interpret as zsh script
#SBATCH -p wacc                        # specify the partition
#SBATCH --job-name=MyAwesomeJob
#SBATCH --output=job_output-%j.txt     # output file: "%j" represents the different job number
#SBATCH --time=0-00:01:00              # time limit, default 60 minutes
#SBATCH --nodes=2                      # number of nodes (see pic below)
#SBATCH --ntasks=16                    # number of tasks without defining number of nodes (see pic below)
#SBATCH --ntasks-per-node=8            # we don't need all three (nodes, ntasks, ntasks-per-node)
#SBATCH --cpus-per-task=4              # number of cores for multi-threaded jobs

#SBATCH --gres=gpu:1                   # request for Generic RESource: a specific number of things, such as graphics cards 
#SBATCH --constraint=haswell           # just a hint to the scheduler and does not guarantee any particular resource will be allocated

cd $SLURM_SUBMIT_DIR                   # Slurm invokes the script from your home directory

date                                   # capture output of many commands
hostname
mpirun awesome_MPI_CUDA_program -np $SLURM_NTASKS

result=$?                              # normal shell script
if [ $result -ge 0 ]; then
    echo “mpirun in job $SLURM_JOB_ID reported error upon finishing”
    exit $result
fi
```
<img width="727" alt="image" src="https://user-images.githubusercontent.com/84046974/191062782-5d00e455-e514-4967-82f3-9d98ec3a0049.png">
<img width="723" alt="image" src="https://user-images.githubusercontent.com/84046974/191062991-ac7fc20b-4cd7-4d1b-a447-5ea0a322f581.png">

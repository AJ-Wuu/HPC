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

## Hardware-Software Interplay
### Instruction & Data
* Instruction: stored in memory as string of 0 & 1 bits (often 32 or 64 bits)
* Piece of Data: stored in memory as string of 0 & 1 bits (8 bits, or 16, or 64, etc. – can be even user defined)
* In terms of storage and movement, there is no distinction between instructions and data
  * Instructions are fetched & decoded & executed (more on this later)
  * Data is used to produce results according to rules specified by the instructions
#### CU (Control Unit)
* The CU controls the DATAPATH (the hardware collection of functional units + registers + buses)
* Duties
  * Bringing in the next instruction from memory
  * Decoding the instruction (possibly breaking it up into uops -- micro-operations)
  * Engaging in actions to enable “instruction level parallelism” (ILP, more later)
* CU manages/coordinates based on information encoded in the specific instruction it works on
* A limited number of prompts that the CU understands and thus act upon
#### ALU (Arithmetic Logic Unit)
* Made up of a bunch of so called “execution units” or “functional units”
* In charge of executing arithmetic and load/store operations
#### ISA (Instruction Set Architecture)
* Define a “vocabulary” subsequently used by the compiler to specify the actions of a processor
* The same line of C code can lead to a different set of instructions on two different processors (CPUs) because two CPUs might implement two different ISAs
* RISC & CISC
  * Reduced Instruction Set Computing Architecture
    * An instruction coded into a fixed set of bits (used to be 32, now it’s 64)
    * Instructions operate on registers
    * Simpler to comprehend, provision for, and work with
  * Complex Instruction Set Computing Architecture
    * Instructions have various lengths
    * Flexibility at the price of complexity and power
![image](https://user-images.githubusercontent.com/84046974/191065490-31c1eb91-8612-4ec9-97a8-2a0bc1083764.png)
![image](https://user-images.githubusercontent.com/84046974/191066153-a1beda39-7d31-4b38-a1ba-d6f775ca726e.png)

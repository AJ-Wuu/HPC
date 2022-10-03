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
<img width="800" alt="image" src="https://user-images.githubusercontent.com/84046974/191062782-5d00e455-e514-4967-82f3-9d98ec3a0049.png">
<img width="800" alt="image" src="https://user-images.githubusercontent.com/84046974/191062991-ac7fc20b-4cd7-4d1b-a447-5ea0a322f581.png">

## Pipelining
* Important Remark: Pipelining does not decrease the time to complete one **instruction** but rather it increases the throughput of the processor by overlapping different stages of processing of different instructions
## Memory Mountain
* ```volatile``` keyword forces to write the variable to main memory
## Virtual Memory
* malloc() & calloc()
  * malloc() doesn't do initialize, so it can tolerate memory requirement exceeding space in physical memory
  * calloc() does initialize, so it needs the memory to be prepared instantly -- no tolerance for exceeding memory requirement
  * calloc() takes longer as physical mem actually provisioned and also set to zero

## CUDA (Compute Unified Device Architecture)
* GPU is a highly multithreaded co-processor
* Differences between GPU and CPU threads:
  * GPU threads are extremely lightweight and very little creation overhead
  * GPU needs 1000s of threads for full efficiency; Multi-core CPU needs only a few heavy ones
* Data needs to be copied into GPU device memory and the results need to be fetched back
* For GPU computing to pay off, the data transfer overhead should be overshadowed by the GPU number crunching that draws on that data
* The CUDA kernel calls and copying to/from GPU are managed by the CUDA runtime in a separate stream associated w/ the GPU execution
* The CUDA runtime places all calls that invoke the GPU in a stream (i.e., ordered collection) of calls (FIFO stream)
* Asynchronicity between host and device:
  * Host: continuing execution right away after launching a kernel
  * Device: taking on the next task in the sequence of tasks in the stream
```
#include <cuda.h>
#include <iostream>

__global__ void simpleKernel(int* data) {
    //this adds a value to a variable stored in global memory
    data[threadIdx.x] += 2*(blockIdx.x + threadIdx.x);
}

int main() {
    const int numElems = 4;
    int hostArray[numElems], *devArray;

    //allocate memory on the device (GPU); zero out all entries in this device array 
    cudaMalloc((void**)&devArray, sizeof(int) * numElems);
    cudaMemset(devArray, 0, numElems * sizeof(int));

    //invoke GPU kernel, with one block that has four threads
    simpleKernel<<<1,numElems>>>(devArray);
    
    //bring the result back from the GPU into the hostArray 
    cudaMemcpy(&hostArray, devArray, sizeof(int) * numElems, cudaMemcpyDeviceToHost);

    //print out the result to confirm that things are looking good 
    std::cout << "Values stored in hostArray: " << std::endl;
    for (int i = 0; i < numElems; i++)
        std::cout << hostArray[i] << std::endl;
    
    //release the memory allocated on the GPU 
    cudaFree(devArray);
    return 0;
}
```

## GPU Execution Configuration
* The HOST(the master CPU thread) instructs the DEVICE (the GPU card) with the number of threads to execute a KERNEL
* A kernel function must be called with an execution configuration
  * Threads in a block
    * can be organized as a 3D structure (x,y,z)
    * maximum x- or y-dimension of a block is 1024
    * maximum z-dimension of a block is 64
    * maximum number of threads in each block is 1024
  * Blocks in a grid
    * can be organized as a 3D structure
    * max of 2<sup>31</sup>-1 arranged as 65,535 by 65,535
  * if more threads are needed -- call the kernel again
```c++
__global__void kernelFoo(...); // declaration
dim3DimGrid(100, 50);        // 2D grid structure, w/ total of 5000 thread blocks
dim3DimBlock(4, 8, 8);       // 3D block structure, with 256 threads per block
kernelFoo<<<DimGrid, DimBlock>>>(...algorithms...);
```

## Overshoot
* **```const int blocksPerGrid = (arraySize + threadsPerBlock – 1) / threadsPerBlock;```**
* M = number of threads
* 1D CUDA Block: ```int index = threadIdx.x + blockIdx.x * M;```

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

## Hardware-Software Interplay
### Instruction & Data
* Instruction: stored in memory as string of 0 & 1 bits (often 32 or 64 bits)
* Piece of Data: stored in memory as string of 0 & 1 bits (8 bits, or 16, or 64, etc. – can be user defined)
* In terms of storage and movement, there is no distinction between instructions and data
  * Instructions are fetched & decoded & executed
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
<img width="800" alt="image" src="https://user-images.githubusercontent.com/84046974/191065490-31c1eb91-8612-4ec9-97a8-2a0bc1083764.png">
<img width="800" alt="image" src="https://user-images.githubusercontent.com/84046974/191066153-a1beda39-7d31-4b38-a1ba-d6f775ca726e.png">

### Fetch-Decode-Execute (FDX) cycle -- keeps the CU and ALU busy, instruction after instruction
* Fetch: an instruction is fetched from memory
* Decode: the string of 1s and 0s are decoded by the CU
* Execute: once all data (operands) available, instruction is executed (can take several cycles of the CPU)
### Decoding Instruction Types
#### Opcode contains which type it is
#### Type I (immediate)
| op (6 bits) | rs (5 bits) | rt (5 bits) | constant or address (16 bits) == the “immediate” value == the offset value in instructions |
|:-----------:|------------:|------------:|-------------------------------------------------------------------------------------------:|
#### Type R (register)
| op (6 bits) | rs (5 bits) | rt (5 bits) | rd (5 bits) == new dest | shamt (5 bits) == shift amount | funct (6 bits) == function code |
|:-----------:|------------:|------------:|------------------------:|-------------------------------:|--------------------------------:|
#### Type J (jump)
| op (6 bits) | address (26 bits) == word address, not an offset |
|:-----------:|-------------------------------------------------:|
### Transistors: AND, OR, NOT
<img width="800" alt="image" src="https://user-images.githubusercontent.com/84046974/191073967-967f724f-aa78-473f-992d-112c6143bbf4.png">

### Registers
* A hardware asset whose role is that of information storing (information: data value or instruction)
* The storage type with shortest latency – part of the CU & ALU combo
  * Latency – can be of the order of hundreds of picoseconds
* Size = 32 bits (now 64 bits)
* Number of registers = 32 -> depending on the design of the chip
* Types
  * Staple in most ISA
    * Current Instruction Register (CIR) – holds the instruction that is executed
    * Program Counter (PC) – holds address of the instruction executed next
    * Memory Data Register (MDR) – holds data read in from memory or, alternatively, produced by the ALU and waiting to be stored in memory
    * Memory Address Register (MAR) – holds address of RAM memory location where input/output data is supposed to be read in/written out
    * Return Address (RA) – the address where upon finishing a sequence of instructions, the execution should jump and commence with the execution of subsequent instruction
  * Other saved between function calls
    * Registers for Subroutine Arguments (4) – a0 through a3
    * Registers for Temporary Variables (10) – t0 through t9
    * Registers for Saved Temporary Variables (8) – s0 through s7
  * Other involved in handling function calls
    * Stack Pointer (sp) – a register that holds an address that points to the top of the stack
    * Global Pointer (gp) –  register that holds on to a global pointer that points into the middle of a 64KB block of memory in the heap that holds constants and global variables (these objects can be quickly accessed with a single load or store instruction)
    * Frame Pointer (fp) - a register that holds an address that points to the beginning of the procedure frame (for instance, the previous sp before this function changed its value)
### Pipelining
#### FDX - Each stage has substages and takes a certain number of cycles
Stage 1: Fetch an instruction  
Stage 2: Decode the instruction  
Stage 3: Data access  
Stage 4: Execute the operation (e.g., might be a request to calculate an address)  
Stage 5: Write-back into register file
#### Characteristics
* Balanced == each of pipeline stage takes the same amount of time for completion
* Benefits: Faster execution & No code rewrite needed
* One clock cycle: the amount of time required to complete one stage of the pipeline
* Pipelined processor: one instruction retired at each clock cycle
* Non-pipelined processor: no overlap when executing instructions
* Important Remark: Pipelining does not decrease the time to complete one **instruction** but rather it increases the throughput of the processor by overlapping different stages of processing of different instructions
#### Hazards
1. Structural Hazards
    1. Increase register number
    2. Introduce bubbles in the pipeline to slow down the execution
    3. OOOE (Out-Of-Order Execution)
2. Data Hazards
    1. Immediate Result Forwarding: provide the means for that value in the ALU to be made available to other stages of the pipeline right away
        1. avoids bubbling
        2. however, this does not guarantee resolution
    2. OOOE
3. Control Hazards
    1. Static Branch Prediction: several heuristics for proceeding, like a do-while construct
    2. Dynamic Branch Prediction: At a branching point, the branch/no-branch decision can change during the life of a program based on recent history
        1. In some cases, branch prediction accuracy hits 90%
        2. Use with if-then-else statements
    3. Note: when your prediction is wrong you have to discard the [micro]instruction[s] executed speculatively and take the correct execution path
### Multiple-Issue
* Superscalar == a chip that can do multiple-issue
* A multiple-issue processor core has the hardware chops to issue more than one instruction per cycle
* On average, more than one instruction is processed by the same core in the same clock cycle
* Static multiple-issue: predefined, doesn’t change at run time
* Dynamic multiple-issue: determined at run time, the chip has dedicated hardware resources that can identify and execute additional work
  * Make sure the result has no difference if the instructions are executed "multiple-issue" or "single issue"
<img width="200" alt="image" src="https://user-images.githubusercontent.com/84046974/191537755-204e4185-3fe8-4e1d-8986-9009c8c916fc.png" align="left">
<img width="300" alt="image" src="https://user-images.githubusercontent.com/84046974/191537857-d4da20a6-f9ce-4b33-b92d-36736c7db4b3.png">
<img width="400" alt="image" src="https://user-images.githubusercontent.com/84046974/192000936-ee11b2d7-6e66-459e-af0a-383a9f2f1748.png">

### Parallelism happens on a SINGLE hardware CORE
* ILP (Instruction-Level Parallelism)
  * simultaneously multiple instructions belong to the same program or process
  * associated with one PC (Program Counter)
* TLP (Thread Level Parallelism)
  * simultaneously instructions from different processes or different threads
  * associated with multiple PCs
* Note that superscalar is associated with one PC
* Process & Thread
  * Processes run in separate memory spaces -- distributed-memory parallel computing
  * Threads (of the same process) run in a shared memory space -- shared-memory parallel computing
### Multi-threading
* Coarse: Instruction Control switches to issuing from another thread only when the first thread runs into a long-latency operation 
* Fine: Instruction Control can issue instructions from a different thread each cycle
* Simutaneous: Instruction Control can issue instructions from different threads in the same cycle
### Measure Speed of Execution
* 𝐶𝑃𝑈 𝐸𝑥𝑒𝑐𝑢𝑡𝑖𝑜𝑛 𝑇𝑖𝑚𝑒 = 𝐼𝑛𝑠𝑡𝑟𝑢𝑐𝑡𝑖𝑜𝑛 𝐶𝑜𝑢𝑛𝑡 × 𝐶𝑃𝐼 × 𝐶𝑙𝑜𝑐𝑘 𝐶𝑦𝑐𝑙𝑒 𝑇𝑖𝑚𝑒 
* 𝐶𝑃𝑈 𝐸𝑥𝑒𝑐𝑢𝑡𝑖𝑜𝑛 𝑇𝑖𝑚𝑒 = (𝐼𝑛𝑠𝑡𝑟𝑢𝑐𝑡𝑖𝑜𝑛 𝐶𝑜𝑢𝑛𝑡 × 𝐶𝑃𝐼) / 𝐶𝑙𝑜𝑐𝑘 𝑅𝑎𝑡𝑒
* Improve performance
  * reducing Clock Cycle Time hits the power wall
  * reducing the Instruction Count (IC) often leads to an increase in CPI, and vice versa

## Memory Aspects
### SRAM & DRAM
* SRAM – Static Random Access Memory
  * One element == a special circuit, called flip-flop
  * One flip-flop requires four to six transistors
  * Each of these elements stores one bit of information
  * Static == once set, the element stores the value set as long as the element is powered
* DRAM - Dynamic Random Access Memory
  * information stored as a charge in a **capacitor**
  * Drawback: capacitors leak -- need to recharge often
<img width="800" alt="image" src="https://user-images.githubusercontent.com/84046974/191094139-41bd2bcf-7dac-406a-af2f-971074bd651a.png">
<img width="800" alt="image" src="https://user-images.githubusercontent.com/84046974/191094173-5a345b99-c407-4056-bbad-b856e32ae460.png">

### Cache
* All caches are typically on the chip and are SRAM
* All read/write operations go through cache
* Caches used to hold both data and instructions
* Chip datapath:
<img width="800" alt="image" src="https://user-images.githubusercontent.com/84046974/191095052-6f381f08-91b1-4afb-9c1d-a79fa8fd9b0f.png">
<img width="800" alt="image" src="https://user-images.githubusercontent.com/84046974/191095414-8eb9aa17-d9c9-407e-9d96-01dd83c3f93c.png">
<img width="800" alt="image" src="https://user-images.githubusercontent.com/84046974/191095909-c535e491-1203-4e78-bed8-3725a753711b.png">

### Principle of Locality
* Temporal locality: Recently referenced items are likely to be referenced again in the near future
* Spatial locality: Items with nearby addresses tend to come into use together
### Cache Line / Cache Block
* Policy
  * Placement policy: determines where Block b goes
  * Replacement policy: determines which block gets evicted (victim), should there be any freedom of choice
  * Types
    * Direct mapping (E = 1, fast, sometimes need to evict useful parts as no freedom at all): one apartment per floor, like a skinny skyscraper
    * Associative, K-way (E = k, decent compromise): a certain number of floors, each with K apartments, like a Hilton Garden Inn
    * Fully associative cache (S = 1, full-control, high overhead in finding data and deciding whom to evict): apartment complex with ground level only, like a Motel
* Cache Miss Reasons
  * cold / compulsory miss: cache is empty
  * capacity miss: not enough space, working set is too large
  * conflict miss: enough space but "bad luck" -- multiple data objects map to the same block
* Cache Miss by Request Types
  * read miss from instruction cache
  * read miss from data cache
  * write miss to data cache
* General Cache Organization - B × E × S = T
  * B: number of bytes in a cache line (typically 64)
  * E: a set of cache lines (typically 1, 2, 4, 8, or 16 -> 2^k)
  * S: the number of sets that make up the cache
  * T: total cache size
<img width="800" alt="image" src="https://user-images.githubusercontent.com/84046974/191102155-a87e2842-3258-45a3-aec1-237d220125c5.png">

* Write to cache
  * Hit
    * Write-through (write immediately to main memory as well)
    * Write-back (defer write to memory until replacement of line) -> Need a dirty bit (indicate whether line different from memory or not)
  * Miss
    * Write-allocate (write to next level down memory, but also update this cache level)
    * No-write-allocate (writes straight to next level down memory, do not load into this cache level)
  * Typical combos met in practice:
    * Write-back + Write-allocate -> more common
    * Write-through + No-write-allocate
* Hit & Miss
  * Hit Time: 4 clock cycles for L1, 10 clock cycles for L2
  * Miss Penalty: typically 50-200 cycles for main memory
### Memory Mountain
* Measures read-throughput as a function of spatial and temporal locality == Compact way to characterize memory system performance
* read throughput == read bandwidth == number of bytes read from main memory per second (MB/s)
* effective bandwidth == computed by dividing amount of data moved to solve the problem (in Bytes) to the amount of time required to finish the execution (in seconds)
  * effective bandwidth <= the bandwidth
* ```volatile``` keyword forces to write the variable to main memory
### Virtual Memory
1. Virtual memory (for compilers): a fictitious space of 2<sup>32</sup> addresses (on 32 bit architectures) in which a process sees its data being placed, the instructions stored, etc.
    1. Pointers all point to virtual memory
    2. Kernel development deals with physical memory
    3. malloc() & calloc()
        1. malloc() doesn't do initialize, so it can tolerate memory requirement exceeding space in physical memory
        2. calloc() does initialize, so it needs the memory to be prepared instantly -- no tolerance for exceeding memory requirement
        3. calloc() takes longer as physical mem actually provisioned and also set to zero
2. Physical memory: a busy place that hosts at the same time data and instructions associated with tens of applications running on the system
3. MMU (Memory Management Unit) -- functionality
    1. Virtual Memory management
    2. Memory protection
    3. Cache control
    4. Bus arbitration 
<img width="800" alt="image" src="https://user-images.githubusercontent.com/84046974/191547499-a789a528-3c29-4c15-812d-49703eef438e.png">
<img width="800" alt="image" src="https://user-images.githubusercontent.com/84046974/191547766-fd7ad873-7bcd-4790-84b5-ebe2a4855ff3.png">

#### Page Table (PT) -- stored in main memory
* Virtual memory space is mapped back into the physical memory through a Page Table
* Not EVERY virtual address is mapped to a physical memory address -- it works like **a page of virtual memory mapped to a frame of physical memory**
  * The size of a page (or frame) is typically 4096 bytes = 2<sup>12</sup>
  * Compare to 64 bytes, the size of a cache line (aka memory block)
<img width="800" alt="image" src="https://user-images.githubusercontent.com/84046974/191548140-2f4d8749-12e5-4a4e-952d-597499383f39.png">

* clean / dirty bit (each entry in the PT has **at least** one): indicates that translation to frame should not proceed immediately
  * clean: the corresponding frame is loaded in main memory + the very frame has not been associated with other virtual page
  * dirty: page fault + OS need to look for the needed frame on disk (secondary memory)
    * a lot of such bits will be dirty when you start the execution of a program as the data & instructions are not in memory

#### Translation Lookaside Buffer (TLB) == a “cache” for the address translation process
* TLB holds the translation of a small collection of virtual page numbers into frame IDs
* A TLB miss leads to substantial overhead in the translation of a virtual memory address
* In general,
  * Good: hit == quicker address translation
  * Bad:
    * miss -> go to main memory -- required to read the PT entry
    * miss again at main memory -> go to secondary memory -- **page fault** == significant latency & **memory thrashing** == repeatedly hitting page faults that time wasted in handling page faults dominates all the rest
<img width="800" alt="image" src="https://user-images.githubusercontent.com/84046974/191551409-654e1b3b-9806-4343-9985-712d1beeaa2a.png">

#### Frame
* Refresher: a memory page, or virtual page, is a fixed-length contiguous block of virtual memory, described by a single entry in the Page Table
* Large Size
  * Pros: less TLB pressure
  * Cons: waste of physical memory, bad for other processes with time-sharing (multi-tasking)

## Parallel Computing
### Main causes for sequential computing speed stalling
#### Memory Wall -- the growing disparity of speed between CPU and memory outside the chip
* Reason: latency and limited communication bandwidth beyond chip boundaries
* latency = how many seconds it takes a packet of data to get from one designated point to another (measured by sending 32 bits data in memory transaction)
  * tough
  * like build a high-speed train & rails
* bandwidth = how much data can be transferred per second
  * doable -- add more "pipes"
  * like adding more cars to a train
* Memory-wise, you may or may not be at top speed for your memory
  * Effective bandwidth == Nominal bandwidth: you move lots of data
  * Effective bandwidth < Nominal bandwidth: you bring a block of data, use little of it, and ask for data from a different block
<img width="600" alt="image" src="https://user-images.githubusercontent.com/84046974/191994937-aa892d88-7b83-4bd2-b14c-67c5a1eec998.png" align="left">
<img width="350" alt="image" src="https://user-images.githubusercontent.com/84046974/191994961-2819c7bc-ee7c-4764-8900-44b2ab862c2f.png">

#### Instruction Level Parallelism (ILP) Wall
* Dedicated hardware speculatively executes future instructions before the results of current instructions are known, while providing hardware safeguards to prevent the errors that might be caused by out of order execution 
* Branches must be “guessed” to decide what instructions to execute simultaneously 
* Data dependencies may prevent successive instructions from executing in parallel, even if there are no branches

#### Power Wall
* Power dissipation related to the clock frequency and feature length
  * clock rates are limited to some threshold value
* Significant increase in clock speed without heroic/expensive cooling not possible

### Dennard scaling
* dictates how the voltage/current, frequency should change in response to smaller feature size
* voltage getting lower and lower
* static power losses more prevalent
* The bad news: direct tunneling gate leakage current (thin dielectric) ⨁ transistors too close
* Amount of power dissipated grows to high levels
  * Thermal runaway 
  * Moving towards threshold at which computing not reliable 
### Analogy
* Ox: one powerful CPU core, running at high frequency, large cache to hide memory latency
* Chicken: many meagre processors, running at lower frequency, not much cache or CU brain, requires other ways to hide memory latency

### Flynn’s Taxonomy of Architectures
#### SISD - Single Instruction/Single Data & SIMD - Single Instruction/Multiple Data
* vector length = 8 ints × 4 bytes each × 8 bits/byte = 256 bit vector length
<img height="200" alt="image" src="https://user-images.githubusercontent.com/84046974/192000629-759288f8-466b-4a33-bb35-07331d00f1d9.png" align="left">
<img height="200" alt="image" src="https://user-images.githubusercontent.com/84046974/192001359-db164242-b9a6-4e36-8a3a-5244e4fac8a8.png">

#### MISD - Multiple Instruction/Single Data & MIMD - Multiple Instruction/Multiple Data
<img height="200" alt="image" src="https://user-images.githubusercontent.com/84046974/192002192-88f06b75-1a68-4529-959e-ce78471f7504.png" align="left">
<img height="200" alt="image" src="https://user-images.githubusercontent.com/84046974/192002280-beb0d8da-0a29-4adf-9b39-9905de4137ea.png">

#### Comparison
<img height="200" alt="image" src="https://user-images.githubusercontent.com/84046974/192001508-8cd17b36-5047-49e1-9670-eb893f7dc0ed.png" align="left">
<img height="200" alt="image" src="https://user-images.githubusercontent.com/84046974/192002363-31b3e2c6-2f08-4960-801b-67015e931cd2.png">

### Amdahl’s Law - law of diminishing returns
* illustrate how going parallel with a part of your code is going to lead to overall speedups

### Algorithmic vs. Intrinsic Scaling
* Algorithmic -> Scaling of a solution algorithm
  * strictly about a mathematical solution algorithm
  * expresses how the algorithm scales with the problem size
* Implementation -> Scaling of a solution on a certain architecture
  * Intrinsic Scaling: how the execution time changes with an increase in the size of the problem
  * Strong Scaling: how the execution time changes when you increase the processing resources
  * Weak Scaling: how the execution time changes when you increase the problem size but also the processing resources in a way that basically keeps the ratio of “problem size/processor” constant
* Important: if Intrinsic Scaling significantly worse than Algorithmic Scaling then probably memory transactions are dominating the implementation
* If the problem doesn’t scale, it can be an interplay of several factors
  * the intrinsic nature of the problem at hand
  * the attributes (organization) of the underlying hardware
  * the algorithm used to solve the problem; i.e., the parallelism it exposes


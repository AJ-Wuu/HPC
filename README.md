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
![image](https://user-images.githubusercontent.com/84046974/191073967-967f724f-aa78-473f-992d-112c6143bbf4.png)
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
* Important Remark: Pipelining does not decrease the time to complete one instruction but rather it increases the throughput of the processor by overlapping different stages of processing of different instructions
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
![image](https://user-images.githubusercontent.com/84046974/191094139-41bd2bcf-7dac-406a-af2f-971074bd651a.png)
![image](https://user-images.githubusercontent.com/84046974/191094173-5a345b99-c407-4056-bbad-b856e32ae460.png)

### Cache
* All caches are typically on the chip and are SRAM
* All read/write operations go through cache
* Caches used to hold both data and instructions
* Chip datapath:
![image](https://user-images.githubusercontent.com/84046974/191095052-6f381f08-91b1-4afb-9c1d-a79fa8fd9b0f.png)

### Processor Clock & Memory Clock
<img width="523" alt="image" src="https://user-images.githubusercontent.com/84046974/191095414-8eb9aa17-d9c9-407e-9d96-01dd83c3f93c.png">

### Cache Hierarchy
<img width="161" alt="image" src="https://user-images.githubusercontent.com/84046974/191095638-d8e58031-234a-4231-812f-dd6ce246f6dc.png">
<img width="397" alt="image" src="https://user-images.githubusercontent.com/84046974/191095909-c535e491-1203-4e78-bed8-3725a753711b.png">

### Principle of Locality
* Temporal locality: Recently referenced items are likely to be referenced again in the near future
* Spatial locality: Items with nearby addresses tend to come into use together


# Hardware-Software Interplay
## Instruction & Data
* Instruction: stored in memory as string of 0 & 1 bits (often 32 or 64 bits)
* Piece of Data: stored in memory as string of 0 & 1 bits (8 bits, or 16, or 64, etc. – can be even user defined)
* In terms of storage and movement, there is no distinction between instructions and data
  * Instructions are fetched & decoded & executed (more on this later)
  * Data is used to produce results according to rules specified by the instructions
### CU (Control Unit)
* The CU controls the DATAPATH (the hardware collection of functional units + registers + buses)
* Duties
  * Bringing in the next instruction from memory
  * Decoding the instruction (possibly breaking it up into uops -- micro-operations)
  * Engaging in actions to enable “instruction level parallelism” (ILP, more later)
* CU manages/coordinates based on information encoded in the specific instruction it works on
* A limited number of prompts that the CU understands and thus act upon
### ALU (Arithmetic Logic Unit)
* Made up of a bunch of so called “execution units” or “functional units”
* In charge of executing arithmetic and load/store operations
### ISA (Instruction Set Architecture)
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

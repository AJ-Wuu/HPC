# Spacial Locality - task3.f Observation
## Data got from Euler
1024  
2752.59  
258.144  
776.942  
258.144  
11140.6  
258.144  
2782.85  
258.144

## Reflection
### difference in the times for mmul1, mmul2 and mmul3
In general, we have mmul2 << mmul1 << mmul3. Specifically, mmul1 takes almost 4 times of mmul2, and mmul3 takes almost 4 times of mmul1.

### explaination of this performance result
First, we notice that A, B and C are all stored in row-major order. This leads to the consequence that with spacial locality, when reading in an element of a row, the whole row will be taken into the cache. This always gets us more efficiency in accessing row data than column data.
Then, we analyze the accessing pattern within each function. With one cell of C, mmul1 goes row-by-row with A and column-by-column with B. With one cell of A, mmul2 goes row-by-row with B and C. With one cell of B, mmul3 goes row-by-row with A and C.
Hence, we could see that mmul2 takes the most advantage of caching rows, so it's the fastest; while mmul3 takes the least advantage of caching rows, so it's the slowest.

### difference or similarity between mmul1 and mmul4
The time consumption is almost the same comparing mmul1 and mmul4. The only difference is mmul1 using pointers and mmul4 using vectors. However, the difference in data structure doesn't affect how the caches & memories dealing with the computation, so this may not bring much difference.

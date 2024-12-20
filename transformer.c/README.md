# Performance Measurements

The token per second metric is measure by generating 100 tokens for 10 runs and averaging the results.  
The performance is measure on an Intel Core i7-1165G7 - 2.80GHz CPU with 8 Threads (4 Physical + 4 HyperThreads)  

The CPU supports AVX512 SIMD  

For reference, average tok/s using HuggingFace is 30.64  

## GPT2 Small - 137M Parameters

| Experiment Name | Token/s |
|-----------------|---------|
| Single Threaded    |   5.47      |
| Single Threaded: Inline Functions    |   5.52      |
| Multi Threaded: Parallel MHA    |   6.09      |
| Multi Threaded: Parallel MHA, Parallel Matmul    |   17.77      |
| Multi Threaded: Parallel MHA, Parallel Matmul, SIMD    |   36.6      |
| Multi Threaded: Parallel MHA, Parallel Matmul, SIMD, Hinting at Largest SIMD Registers    |   43.17      |
| Multi Threaded: Parallel MHA, Parallel Matmul, SIMD, Hinting at Largest SIMD Registers, Optimize Memory Accesses    |   51.72      |


## GPT2-xl - 1.61B Paramters

For reference, HuggingFace GPT2-xl performs at 3.12 tok/s  

| Experiment Name | Token/s |
|-----------------|---------|
| Multi Threaded: Parallel MHA, Parallel Matmul, SIMD, Hinting at Largest SIMD Registers    |   3.88      |
| Multi Threaded: Parallel MHA, Parallel Matmul, SIMD, Hinting at Largest SIMD Registers, Optimize Memory Accesses     |   4.58      |


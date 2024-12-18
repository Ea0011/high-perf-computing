# Performance Measurements

The token per second metric is measure by generating 100 tokens for 10 runs and averaging the results.  
The performance is measure on an Intel Core i7-1165G7 - 2.80GHz CPU with 8 Threads (4 Physical + 4 HyperThreads)  

| Experiment Name | Token/s |
|-----------------|---------|
| Single Threaded    |   5.47      |
| Single Threaded: Inline Functions    |   5.52      |
| Multi Threaded: Parallel MHA    |   6.09      |
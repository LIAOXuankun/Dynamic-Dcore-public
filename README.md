
## Description
Welcome to the readme file of the code for the paper titled **Accelerating D-core Maintenance over Dynamic Directed Graphs**. In this work, we provide a formal definition of the problem of D-core maintenance over dynamic directed graphs. With the proposed theorems as a basis, we initially introduce a local-search-based algorithm along with three optimizations to reduce the search space. Additionally, we propose an H-index-based algorithm to handle batch updates.

## Contributions
- Fundamental problem of D-core maintenance over dynamic directed graphs
- Theorems to reduce the search space
- A local-search-based algorithm that incorporates three optimizations
- An H-index-based algorithm with non-trivial edge-grouping strategies

## Environment
All of our algorithms are implemented in C++ 11 and compiled with the g++ compiler at -O3 optimization level. The development environment used for implementing and testing is:

-Linux version: Oracle Linux 8.6

-g++ version: 11.2.0


## Dataset format

The input are three files, i.e., graph.txt, graph-insert.txt, and graph-delete.txt. graph.txt is the original graph file. graph-insert.txt and graph-delete.txt are the edges to be inserted and deleted, respectively. All of the three files have two columns in the form of "u v". An example is shown below:

```bash
0 1
0 3
1 0
1 2
2 1
2 3
2 7
3 0
4 3
5 2
6 0
6 3
6 4
7 5
2 4
3 7
```

## How to compile the algorithms

To compile the corresponding code, first go to the directory of the code files, then:

```bash
./compile.sh
```
After compilation, executable files named ```dcore``` will be generated.

## Parameters

There are sevaral paramaters to ```dcore```, which are listed below.

```bash
argv[1]: batch_size: the number of edges to be inserted or deleted
argv[2]: input_file_name: e.g., email
argv[3]: thread_number
argv[4]: bool_optimization_h_index: a boolean variable, always set to true, engineering optimization
argv[5]: bool_batch_kmax: a boolean variable, whether use the edge group strategy 1 to group edges
argv[6]: bool_batch_homo_edge_group: a boolean variable, whether use the edge group strategy 2 to group edges
argv[7]: bool_decomp_maintenance: a boolean variable, whether run the SOTA d-core decomposition algorithm
argv[8]: bool_peel_maintenance: a boolean variable, whether run the SOTA d-core maintenance algorithm
argv[9]: bool_k0core_pruning: a boolean variable, whether apply optimization 1
argv[10]: bool_reuse_pruning: a boolean variable, whether apply optimization 3
argv[11]: bool_skip_pruning: a boolean variable, whether apply optimization 2
argv[12]: bool_dfs: a boolean variable, whether run the local-search-based algorithm
```



## How to run the algorithms

An exmaple command to run the SOTA d-core maintenance algorithm, the SOTA d-core decomposition algorithm, and H-index-based algorithm with all the proposed optimizations and edge-grouping strategies, under 16 threads is shown below. Note that if you want to run the algorithm, you should go to the corresponding directory first. Besides, please change the variable ```file_path``` in the main.cpp to your file path before compile our algorithms.

```bash
./dcore 10000 email 16 true true true true true true true true false
```

## 






#ifndef DYNAMIC_DCORE_COMMON_H
#define DYNAMIC_DCORE_COMMON_H

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <bits/stdc++.h>
#include <fcntl.h>
#include <omp.h>

#include <cstdint>
#include <iostream>

#include <string>
#include <vector>
#include <map>
#include <list>
#include <algorithm>
#include <chrono>
#include <set>

#define THE_NUMBER_OF_THREADS 16
#define LMAX_NUMBER_OF_THREADS 16  //for single edge maintenance, parallel maintain multiple lmax value of multiple k-lists


#define KMAX_HIERARCHY false
#define KEDGE_SET false


#define ASSERT(truth) \
    if (!(truth)) { \
      std::cerr << "\x1b[1;31mASSERT\x1b[0m: " \
                << "LINE " << __LINE__ \
                << ", " << __FILE__ \
                << std::endl; \
      std::exit(EXIT_FAILURE); \
    } else

#define ASSERT_MSG(truth, msg) \
    if (!(truth)) { \
      std::cerr << "\x1b[1;31mASSERT\x1b[0m: " \
                << "LINE " << __LINE__ \
                << ", " << __FILE__ << '\n' \
                << "\x1b[1;32mINFO\x1b[0m: " << msg \
                << std::endl; \
      std::exit(EXIT_FAILURE); \
    } else

using namespace std;


#endif //DYNAMIC_DCORE_COMMON_H

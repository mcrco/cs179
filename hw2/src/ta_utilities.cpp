//--------------------------------------------------------------------------
// TA_Utilities.cpp
// Allow a shared computer to run smoothly when it is being used
// by students in a CUDA GPU programming course.
//
// TA_Utilities.cpp/hpp provide functions that programatically limit
// the execution time of the function and select the GPU with the 
// lowest temperature to use for kernel calls.
//
// Author: Jordan Bonilla - 4/6/16
//--------------------------------------------------------------------------

#include "ta_utilities.hpp"

#include <unistd.h> // sleep, fork, getpid
#include <signal.h> // kill
#include <cstdio> // printf
#include <stdlib.h> // popen, pclose, atoi, fread
#include <cuda_runtime.h> // cudaGetDeviceCount, cudaSetDevice
#include <sys/prctl.h>

namespace TA_Utilities
{
  /* Create a child thread that will kill the parent thread after the
     specified time limit has been exceeded */
  void enforce_time_limit(int time_limit) {
      printf("Time limit for this program set to %d seconds\n", time_limit);
      int parent_id = getpid();
      pid_t child_id = fork();
      // The fork call creates a lignering child thread that will 
      // kill the parent thread after the time limit has exceeded
      // If it hasn't already terminated.
      if(child_id == 0) // "I am the child thread"
      {
          // kill child if parent exits first, https://stackoverflow.com/a/36945270
          if (prctl(PR_SET_PDEATHSIG, SIGKILL) == -1) { perror(0); exit(1); }

          sleep(time_limit);
          if( kill(parent_id, SIGTERM) == 0) {
              printf("enforce_time_limit: Program terminated"
               " for taking longer than %d seconds\n", time_limit);
          }
          // Ensure that parent was actually terminated
          sleep(2);
          if( kill(parent_id, SIGKILL) == 0) {
              printf("enforce_time_limit: Program terminated"
               " for taking longer than %d seconds\n", time_limit);
          } 
          // Child thread has done its job. Terminate now.
          exit(0);
      }
      else // "I am the parent thread"
      {
          // Allow the parent thread to continue doing what it was doing
          return;
      }
  } // end "void enforce_time_limit(int time_limit)


} // end "namespace TA_Utilities"

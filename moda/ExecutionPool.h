#pragma once

#include<vector>
#include <mutex>
#include "ExecutionContext.h"

#define MEMORY_POOL_SIZE 1000

namespace moda {
    namespace backend {
        class ExecutionPool {
        public:
            ~ExecutionPool();
            ExecutionPool();
            /// <summary>
            /// Reserves resources for a single execution
            /// </summary>
            /// <param name="reserveSize">Desired size of memory (measured in number of points). The underlying data structure is dynamic, however, in order to save time, it is recommended to initialize enough memory for the algorithm to run without reallocating.
            /// <param name="initialSize">This part of memory will be filled with deep copies of data. Usually filled immideatly after the context creation.</param>
            /// <param name="numberOfObjectives">Size of a point</param>
            /// <param name="contextType">Type of a context</param>
            /// <param name="shallow">If true, this memory block is designed to store only shallow memory copy (in form of pointers)</param>
            /// <returns></returns>
            int reserveContext(int reserveSize, int initialSize, int numberOfObjectives, ExecutionContext::ExecutionContextType contextType, bool shallow = false);
            /// <summary>
            /// Releases context memory
            /// </summary>
            /// <param name="contextId">Id given to the context upon reservation (reserveContext)</param>
            void releaseContext(int contextId);
            /// <summary>
            /// Checks wether there is any space in the context pool.
            /// </summary>
            /// <returns></returns>
            void cleanMemory()
            {
                for (int id : reservedIds)
                {
                    delete contexts[id];
                }
                reservedIds.clear();
                for (int i = 0; i < MEMORY_POOL_SIZE; i++)
                {
                    //contexts[i] = new ExecutionContext(0, 0, 0);
                }
            }
            bool full();
            /// <summary>
            /// Returns a pointer to a context
            /// </summary>
            /// <param name="id">Context id</param>
            /// <returns></returns>
            ExecutionContext* getContext(int id);
        private:

            mutable std::mutex mutex;
            ExecutionContext* contexts[MEMORY_POOL_SIZE];
            std::vector<int> reservedIds;
            static constexpr int MAX_CONTEXTS = MEMORY_POOL_SIZE;

        };
    }
}
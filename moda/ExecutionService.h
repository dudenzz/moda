#pragma once
#include "ExecutionPool.h"
namespace moda {
    namespace backend {
        class ExecutionService {
        public:
            static ExecutionService& getInstance();
            ExecutionPool& getPool();
        private:
            ExecutionService() = default;
            ExecutionService(const ExecutionService&) = delete;
            ExecutionService& operator=(const ExecutionService&) = delete;

            ExecutionPool pool;
        };
    }
}
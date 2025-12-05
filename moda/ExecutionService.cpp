#pragma once
#include "ExecutionService.h"
namespace moda {
    namespace backend {

        ExecutionService& ExecutionService::getInstance()
        {
            static ExecutionService instance;
            return instance;
        }
        ExecutionPool& ExecutionService::getPool() { return pool; }
    }
}
#pragma once
#include "ExecutionContext.h"
namespace moda {
    namespace backend {
        class ObjectiveTransformer {
        public :
            static std::random_device randomDevice;
            static std::mt19937 randomNumberGenerator;
            static void shuffle(ExecutionContext& ctx);
            static void offset(ExecutionContext& ctx, int n);
            static void sort(ExecutionContext& ctx, const std::vector<int>& order);
        };
    }
}
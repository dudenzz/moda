#pragma once
#include "ObjectivesTransformer.h"
namespace moda {
    namespace backend {
        std::random_device ObjectiveTransformer::randomDevice;
        std::mt19937 ObjectiveTransformer::randomNumberGenerator(randomDevice());
        void ObjectiveTransformer::offset(ExecutionContext& ctx, int n) {
            if (n > 0)
                std::rotate(ctx.objectivesOrder.begin(), ctx.objectivesOrder.begin() + n, ctx.objectivesOrder.end());
            if (n < 0)
                std::rotate(ctx.objectivesOrder.rbegin(), ctx.objectivesOrder.rbegin() - n, ctx.objectivesOrder.rend());
        }
        void ObjectiveTransformer::shuffle(ExecutionContext& ctx) {
            std::shuffle(ctx.objectivesOrder.begin(), ctx.objectivesOrder.end(), randomNumberGenerator);
        }
        void ObjectiveTransformer::sort(ExecutionContext& ctx, const std::vector<int>& order) {

            std::sort(ctx.objectivesOrder.begin(), ctx.objectivesOrder.end(), [&](int a, int b) {
                return order[a] < order[b];
                });
        }
    }
}
#include "IQHV.h"
#include <future>
#include "Helpers.h"
#include "Hypervolume.h"
#include "MemoryManager.h"
#include "ExecutionContext.h"
#include "ExecutionPool.h"
#include "ExecutionService.h"
#include "ObjectivesTransformer.h"

#define PARALLEL 1

namespace moda {
	namespace backend {
        DType dummy(DType test, Point& NadirPoint ) { return test+1.0; };
#if CALLBACKS == 1

        DType FDP(int start, int end, int contextId, Point IdealPoint, Point NadirPoint, int recursion, int numberOfObjectives, bool topLevelExecution, clock_t it0, void (*IterationCallback)(int, int, Result*))
#else
        DType FDP(int start, int end, int contextId, Point IdealPoint, Point NadirPoint, int recursion, int numberOfObjectives, bool topLevelExecution)
#endif
        {
            ExecutionService* service = &(ExecutionService::getInstance());
            ExecutionPool* pool = &(service->getPool());

            IQHVExecutionContext* context = (IQHVExecutionContext*)&(*pool->getContext(contextId));
#if UNDERLYING_TYPE == 1
            myvector<Point*>* points;
#elif UNDERLYING_TYPE == 2
            SemiDynamicArray<Point*>* points;
#elif UNDERLYING_TYPE == 3
            SecureVector<Point*>* points;
#else
            std::vector<Point*>* points;
#endif
            int maxIndexUsed = context->maxIndexUsed;
            points = context->points;

            //find pivot
            DType maxContribution = 0;
			int pivotPoint = 0;
            for (int i = start; i < end; i++)
            {
				DType currentContribution = (*points)[i]->contribution(IdealPoint, NadirPoint);
				if (currentContribution > maxContribution)
				{
					maxContribution = currentContribution;
					pivotPoint = i;
				}
            }


            for (int i = start; i < end; i++)
            {
				if (i == pivotPoint)
					continue;
                if (*points->at(i) << *points->at(pivotPoint))
                {

                }
            }

            //subproblems


        }



        
    }
}
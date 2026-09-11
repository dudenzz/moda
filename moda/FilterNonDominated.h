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

        FNDResult FilterNonDominated(int start, int end, std::vector<int> potentiallyDominated, int contextId, Point IdealPoint, Point NadirPoint, int recursion, int numberOfObjectives, bool topLevelExecution, clock_t it0, void (*IterationCallback)(int, int, Result*))
#else
        FNDResult FilterNonDominated(int start, int end, int contextId, Point IdealPoint, Point NadirPoint, int recursion, int numberOfObjectives, bool topLevelExecution)
#endif
        {
            ExecutionService* service = &(ExecutionService::getInstance());
            ExecutionPool* pool = &(service->getPool());

            FDPExecutionContext* context = (FDPExecutionContext*)&(*pool->getContext(contextId));
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
            //Find local best
			Point localBest = *(*points)[start];
            for (int i = start; i < end; i++)
            {
				for (int i = 0; i < numberOfObjectives; i++)
				{
					if ((*points)[i]->ObjectiveValues[i] < localBest.ObjectiveValues[i])
					{
						localBest.ObjectiveValues[i] = (*points)[i]->ObjectiveValues[i];
					}
				}
            }
            //Find local worst
            Point localWorst = *(*points)[start];
            for (int i = start; i < end; i++)
            {
                for (int i = 0; i < numberOfObjectives; i++)
                {
                    if ((*points)[i]->ObjectiveValues[i] > localWorst.ObjectiveValues[i])
                    {
                        localWorst.ObjectiveValues[i] = (*points)[i]->ObjectiveValues[i];
                    }
                }
            }
            //Find pivot
            DType maxContribution = 0;
			int pivotPoint = 0;
            for (int i = start; i < end; i++)
            {
				DType currentContribution = (*points)[i]->contribution(localBest, localWorst);
				if (currentContribution > maxContribution)
				{
					maxContribution = currentContribution;
					pivotPoint = i;
					context->pointStatus[context->localIndexToGlobalIndex[i]] = 1; // pareto
				}
            }
            //FindDominated
            for (int i = start; i < end; i++)
            {
				bool nondominated = true;
				for (int j = 0; j < numberOfObjectives; j++)
				{
					if ((*points)[i]->ObjectiveValues[j] > (*points)[pivotPoint]->ObjectiveValues[j])
					{
                        nondominated = false;
						break;
					}
				}
				if (!nondominated)
				{
					context->pointStatus[context->localIndexToGlobalIndex[i]] = -1; // dominated
				}
            }
            //Decompose
			std::vector<int>* objectivePools = new std::vector<int>[numberOfObjectives];
			for (int objective = 0; objective < numberOfObjectives; objective++)
			{
				for (int i = start; i < end; i++)
				{
					if (i == pivotPoint || context->pointStatus[context->localIndexToGlobalIndex[i]] == -1)
						continue;
					if ((*points)[i]->ObjectiveValues[objective] > (*points)[pivotPoint]->ObjectiveValues[objective])
					{
						objectivePools[objective].push_back(i);
					}
				}
			}
			int totalBoxes = pow(2, numberOfObjectives);
			int bits = numberOfObjectives;
            
			std::map<int, int>* localToGlobalIndexInBoxes = new std::map<int, int>[totalBoxes];
			std::vector<int>* subproblems = new std::vector<int>[totalBoxes];
            for (int i = 0; i < totalBoxes; i++)
            {
				if (i == 0 || i == totalBoxes - 1)
					continue;
				for (int j = 0; j < bits; j++)
				{
					if (i >> j & 1)
					{
						int iter = 0;
						for (int pIndex : objectivePools[j])
						{
							subproblems[i].push_back(pIndex);
							localToGlobalIndexInBoxes[i][iter++] = pIndex;
						}
					}
				} 
            }
			//Build potentially dominated sets *** current problem - the order of ops
			std::vector<int>* potentiallyDominatedBoxes = new std::vector<int>[totalBoxes];
			for (int i = 0; i < totalBoxes; i++)
			{
				if (i == 0 || i == totalBoxes - 1)
					continue;
				for (int additionalBit = 0; additionalBit < bits; additionalBit++)
				{
					if (!(i >> additionalBit & 1))
					{
						for (int pIndex : subproblems[additionalBit])
						{
							potentiallyDominatedBoxes[i].push_back(pIndex);
						}
					}
				}
			}
            //Process boxes
            FNDResult result;
			result.dominatedPoints = std::vector<int>();
			result.paretoFront = std::vector<int>();
            return result;
            //subproblems


        }



        
    }
}
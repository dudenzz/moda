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

        FNDResult FilterNonDominated(std::vector<int> insidePoints, std::vector<int> potentiallyDominated, int contextId, Point IdealPoint, Point NadirPoint, int recursion, int numberOfObjectives, bool topLevelExecution, clock_t it0, void (*IterationCallback)(int, int, Result*))
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
			if (insidePoints.size() == 1)
			{

				context->pointStatus[insidePoints[0]] = 1;
				return FNDResult();
			}
			if (insidePoints.size() == 2)
			{
				context->pointStatus[insidePoints[0]] = 1;
				context->pointStatus[insidePoints[1]] = 1;
				return FNDResult();
			}
            int maxIndexUsed = context->maxIndexUsed;
            points = context->points;
            //Find local best
			Point localBest = *(*points)[insidePoints[0]];
            for (int i : insidePoints)
            {
				for (int j = 0; j < numberOfObjectives; j++)
				{
					if ((*points)[i]->ObjectiveValues[j] > localBest.ObjectiveValues[j])
					{
						localBest.ObjectiveValues[j] = (*points)[i]->ObjectiveValues[j];
					}
				}
            }
            //Find local worst
            Point localWorst = *(*points)[insidePoints[0]];
            for (int i : insidePoints)
            {
                for (int j = 0; j < numberOfObjectives; j++)
                {
                    if ((*points)[i]->ObjectiveValues[j] < localWorst.ObjectiveValues[j])
                    {
                        localWorst.ObjectiveValues[j] = (*points)[i]->ObjectiveValues[j];
                    }
                }
            }
            //Find pivot
            DType maxContribution = -1;
			int pivotPoint = 0;
			std::vector<DType> contributions;
            for (int i : insidePoints)
            {
				DType currentContribution = (*points)[i]->contribution(localBest, localWorst);
				contributions.push_back(currentContribution);
				if (currentContribution > maxContribution)
				{
					maxContribution = currentContribution;
					pivotPoint = i;
					 
				}
            }
			context->pointStatus[pivotPoint] = 1; // pareto
            //FindDominated
            for (int i : insidePoints)
            {
				bool nondominated = false;
				for (int j = 0; j < numberOfObjectives; j++)
				{
					if ((*points)[i]->ObjectiveValues[j] >= (*points)[pivotPoint]->ObjectiveValues[j])
					{
                        nondominated = true;
						break;
					}
				}
				if (!nondominated)
				{
					context->pointStatus[i] = -1; // dominated
				}
            }
            //Decompose
			std::vector<std::vector<int>> subproblems;
			int totalBoxes = pow(2, numberOfObjectives);
			int bits = numberOfObjectives;
			for (int i = 0; i < totalBoxes; i++)
			{
				subproblems.push_back(std::vector<int>());
			}
			for (int i = 0; i < totalBoxes; i++)
			{
				if (i == 0 || i == totalBoxes - 1)
					continue;
				std::vector<int> dominatingBits = std::vector<int>();
				for (int bit = 0; bit < bits; bit++)
				{
					int mask = 1 << bit;
					if (i & mask)
					{
						dominatingBits.push_back(bit);
					}
				}
				for (int j : insidePoints)
				{
					bool belongs = true;
					if (j == pivotPoint) continue;
					for (int bit = 0; bit < bits; bit++)
					{
						if (dominatingBits.end() == std::find(dominatingBits.begin(), dominatingBits.end(), bit))
						{
							if ((*points)[j]->ObjectiveValues[bit] < (*points)[pivotPoint]->ObjectiveValues[bit])
							{
								belongs = false;

							}
						}
						else 
						{
							if (((*points)[j]->ObjectiveValues[bit] >= (*points)[pivotPoint]->ObjectiveValues[bit]))
							{
								belongs = false;
							}
						}
					}
					if (belongs)
					{
						subproblems[i].push_back(j);
					}
				}
			}
			

            

			//Build potentially dominated sets
			std::vector<std::vector<int>> potentiallyDominatedBoxes;
			for (int i = 0; i < totalBoxes; i++)
			{
				potentiallyDominatedBoxes.push_back(std::vector<int>());
			}
			std::vector<std::vector<int>> boxesBySetBits(bits);
			for (int i = 1; i < totalBoxes - 1; i++)
			{
				boxesBySetBits[std::popcount(static_cast<unsigned int>(i))].push_back(i);
			}

			// Iterate sequentially through 1-bit boxes, 2-bit boxes, etc.
			for (int k = 1; k < bits; k++)
			{
				for (int i : boxesBySetBits[k])
				{
					for (int bit = 0; bit < bits; bit++)
					{
						int mask = 1 << bit;
						int potentiallyDominatedBox = i & mask;
						if (potentiallyDominatedBox == i)
							continue;
						for (int dominatedIndex : subproblems[potentiallyDominatedBox])
						{
							potentiallyDominatedBoxes[i].push_back(dominatedIndex);
						}
					}
				}
			}
            //Process boxes
			for (int k = 1; k < bits; k++)
			{
				for (int i : boxesBySetBits[k])
				{
					if (subproblems[i].size() > 0)
					{
						FilterNonDominated(subproblems[i], potentiallyDominatedBoxes[i], contextId, localBest, localWorst, recursion + 1, numberOfObjectives, false, it0, IterationCallback);
					}
				}
			}
            FNDResult result;
			result.dominatedPoints = std::vector<int>();
			result.paretoFront = std::vector<int>();
			int tempIterator = 0;
			for (int status : context->pointStatus)
			{
				if (status == 1)
				{
					result.paretoFront.push_back(tempIterator);
				}
				else if (status == -1)
				{
					result.dominatedPoints.push_back(tempIterator);
				}
				tempIterator += 1;
			}
            return result;
            //subproblems


        }



        
    }
}
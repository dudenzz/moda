#include "IQHV.h"
#include <future>
#include "Helpers.h"
#include "Hypervolume.h"
#include "MemoryManager.h"
#include "ExecutionContext.h"
#include "ExecutionPool.h"
#include "ExecutionService.h"
#include "ObjectivesTransformer.h"
// 0 = switched off
// 1 = switched on
#define PARALLEL 1

/*************************************************************************

 Improved Quick Hypervolume Computation

 ---------------------------------------------------------------------

                           Copyright (C) 2025
          Andrzej Jaszkiewicz <ajaszkiewicz@cs.put.poznan.pl>
          Piotr Zielniewicz <piotr.zielniewicz@cs.put.poznan.pl>
          Jakub Dutkiewicz <jakub.dutkiewicz@put.poznan.pl>

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at https://mozilla.org/MPL/2.0/.

 ----------------------------------------------------------------------

 Relevant literature:

 [1]  Andrzej Jaszkiewicz and Piotr Zielniewicz. 2023. 
      On-line Quick Hypervolume Algorithm. 
      In Proceedings of the 
      Companion Conference on Genetic and Evolutionary Computation (GECCO '23 Companion). 
      Association for Computing Machinery, 
      New York, NY, USA, 371–374. https://doi.org/10.1145/3583133.3590650

 

*************************************************************************/
namespace moda {
	namespace backend {
        DType dummy(DType test, Point& NadirPoint ) { return test+1.0; };
        DType IQHV(int start, int end, int contextId, Point IdealPoint, Point NadirPoint, int recursion, int numberOfObjectives, int outerIteratorValue,  int fullSize)
        {
            ExecutionService* service = &(ExecutionService::getInstance());
            ExecutionPool* pool = &(service->getPool());
    
            IQHVExecutionContext* context = (IQHVExecutionContext*) & (*pool->getContext(contextId));
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

            std::vector<std::future<DType>> threads;
            std::vector<int> memorySlots;

            if (end < start) {
                return 0;
            }

            for (int i = 0; i < numberOfObjectives; i++)
            {
                if (IdealPoint[i] < NadirPoint[i])
                    int k = 0;
            }


            int oldmaxIndexUsed = maxIndexUsed;

            // If there is just one point
            if (end == start) {
                DType totalVolume = Hypervolume(&NadirPoint, (*points)[start], &IdealPoint, numberOfObjectives);

                return totalVolume;
            }

            // If there are just two points
            if (end - start == 1) {
                DType totalVolume = Hypervolume(&NadirPoint, (*points)[start], &IdealPoint, numberOfObjectives);
                totalVolume += Hypervolume(&NadirPoint, (*points)[end], &IdealPoint, numberOfObjectives);
                Point* point2 = new Point(*((*points)[start]));
                unsigned j;
                for (j = 0; j < numberOfObjectives; j++) {
                    if (point2->ObjectiveValues[j] > (*points)[end]->ObjectiveValues[j])
                        point2->ObjectiveValues[j] = (*points)[end]->ObjectiveValues[j];
                }
                totalVolume -= Hypervolume(&NadirPoint, point2, &IdealPoint, numberOfObjectives);
                delete point2; // no delete here causes heavy memory leaks
                return totalVolume;
            }

            int iPivot = start;
            DType maxVolume;
            maxVolume = Hypervolume(&NadirPoint, (*points)[iPivot], &IdealPoint, numberOfObjectives);

            // Find the pivot point
            unsigned i;
            for (i = start + 1; i <= end; i++) {
                DType volumeCurrent;
                volumeCurrent = Hypervolume(&NadirPoint, (*points)[i], &IdealPoint, numberOfObjectives);
                if (maxVolume < volumeCurrent) {
                    maxVolume = volumeCurrent;
                    iPivot = i;
                }
            }
            DType totalVolume = Hypervolume(&NadirPoint, (*points)[iPivot], &IdealPoint, numberOfObjectives);

#ifdef callbacks
            HypervolumeResult tempResult;
            if (recursion == 0)
            {
#ifdef _MSC_VER 
                tempResult.ElapsedTime = clock() - it0;
#else
                tempResult.ElapsedTime = (clock() - it0) / 1000.0;
#endif
                tempResult.HyperVolume = 0.0;
                tempResult.FinalResult = false;
                tempResult.type = Result::Hypervolume;
                IterationCallback(0, numberOfObjectives * numberOfObjectives, &tempResult);
            }
#endif
            // Build subproblems
            int iPos = maxIndexUsed + 1;
            unsigned j;
            int jj;

            int ic = 0;

            Point partNadirPoint = NadirPoint;
            Point partIdealPoint = IdealPoint;
            std::vector<int> sizes(numberOfObjectives, 0);

            ObjectiveTransformer::offset(*context, 1);

            std::vector<int> objectivesOrder = context->objectivesOrder;


            for (jj = 0; jj < numberOfObjectives; jj++) {
                j = objectivesOrder[jj];
                if (jj > 0) {
                    int j2 = objectivesOrder[jj - 1];
                    if (IdealPoint.ObjectiveValues[j2] > (*(*points)[iPivot])[j2])
                        partIdealPoint.ObjectiveValues[j2] = (*(*points)[iPivot])[j2];
                    else
                        partIdealPoint.ObjectiveValues[j2] = IdealPoint.ObjectiveValues[j2];
                    partNadirPoint.ObjectiveValues[j2] = NadirPoint.ObjectiveValues[j2];
                }
                int partStart = iPos;

                DType lVal = IdealPoint.ObjectiveValues[j];
                DType pVal = (*points)[iPivot]->ObjectiveValues[j];
                DType b;
                if (lVal > pVal)
                    b = pVal;
                else
                    b = lVal;
                for (i = start; i <= end; i++) {
                    if (i == iPivot)
                        continue;
                    DType a;
                    if (lVal > (*points)[i]->ObjectiveValues[j])
                        a = (*points)[i]->ObjectiveValues[j];
                    else
                        a = lVal;


                    if (a > b) {
                        (*points)[iPos++] = (*points)[i];
                    }
                }
                int partEnd = iPos - 1;
                context->maxIndexUsed = iPos - 1;

                if (partEnd >= partStart) {
                    if (IdealPoint.ObjectiveValues[j] > (*points)[iPivot]->ObjectiveValues[j])
                        partNadirPoint.ObjectiveValues[j] = (*points)[iPivot]->ObjectiveValues[j];
                    else
                        partNadirPoint.ObjectiveValues[j] = IdealPoint.ObjectiveValues[j];
                    if (PARALLEL == 0 || (partEnd - partStart) < fullSize*0.25)
                    {
                        totalVolume += IQHV(partStart, partEnd, contextId, partIdealPoint, partNadirPoint, recursion + 1, numberOfObjectives, jj, fullSize);
                    }
                    else {
                        int points_to_reserve = 4 * (partEnd - partStart) * pow(2, numberOfObjectives / 2) + 1;
                        int newSlot = service->getPool().reserveContext(points_to_reserve, 0, numberOfObjectives, ExecutionContext::ExecutionContextType::IQHVContext, true);
                        if (newSlot == -1)
                        {
                            totalVolume += IQHV(partStart, partEnd, contextId, partIdealPoint, partNadirPoint, recursion + 1, numberOfObjectives, jj, fullSize);
                        }
                        else
                        {
                            ExecutionPool* pool_inner = &service->getPool();
                            IQHVExecutionContext* newContext = (IQHVExecutionContext*)pool_inner->getContext(newSlot);
                            memorySlots.push_back(newSlot);
                            int no_sol = 0;
                            for (int i = partStart; i <= partEnd; i++)
                            {
                                (*newContext->points)[no_sol++] = (*context->points)[i];
                            }
                            newContext->maxIndexUsed = no_sol - 1;
                            
                            //Point* pnad = new Point(partNadirPoint);
                            //Point* pide = new Point(partIdealPoint);

                            //threads[jj] = std::async(dummy, 1.0, std::ref(pnad));
                            threads.push_back(std::async(IQHV, 0, no_sol - 1, newSlot, partIdealPoint, partNadirPoint, recursion + 1, numberOfObjectives, jj, fullSize));
                            //delete pnad;
                            //delete pide;
                        }
                        //totalVolume += IQHV(0, no_sol - 1, newSlot, partIdealPoint, partNadirPoint, offset, recursion + 1, numberOfObjectives, jj);

                    }

                }


#ifdef callbacks
                if (recursion == 1)
                {
#ifdef _MSC_VER 
                    tempResult.ElapsedTime = clock() - it0;
#else
                    tempResult.ElapsedTime = (clock() - it0) / 1000.0;
#endif
                    tempResult.HyperVolume = totalVolume;
                    tempResult.type = Result::Hypervolume;
                    //IterationCallback(outerIteratorValue * numberOfObjectives + jj + 1, numberOfObjectives * numberOfObjectives, &tempResult);
                }
#endif
            }

            for (std::future<DType>& thread : threads)
            {
                thread.wait();
                totalVolume += thread.get();
            }
            for (int slot : memorySlots)
            {
                service->getPool().releaseContext(slot);
            }

            memorySlots.clear();
            
            if (context->maxIndexUsed > context->maxIndexUsedOverall)
                context->maxIndexUsedOverall = context->maxIndexUsed;
            context->maxIndexUsed = oldmaxIndexUsed;
            
            return (totalVolume);
            //0.520691
        }

        
    }
}
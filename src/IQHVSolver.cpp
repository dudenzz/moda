#include "IQHVSolver.h"
#include "IQHV.h"
#include "MemoryManager.h"
#include "ExecutionContext.h"
#include "ExecutionPool.h"
#include "ExecutionService.h"
namespace moda {
    HypervolumeResult* IQHVSolver::Solve(DataSet* problem, IQHVParameters settings)
    {
        //initialize the problem
        prepareData(problem, settings);
        //call the starting callback
        StartCallback(*currentSettings, "IQHV Solver");
        //initialize an empty result
        HypervolumeResult* r = new HypervolumeResult();
        r->type = Result::ResultType::Hypervolume;
        //calculate the hypervolume
        it0 = clock();
        r->HyperVolume = initAndSolveIQHV( *betterPoint, *worsePoint, currentSettings->nPoints,  settings.callbacks);
        r->ElapsedTime = clock() - it0;
        //std::cout << "\t" << r->ElapsedTime << "\n";
        r->FinalResult = true;
        //call the closing callback
        EndCallback(*currentSettings, r);



        
        delete currentSettings;
        //delete currentlySolvedProblem;
        delete betterPoint;
        delete worsePoint;
        //return the result
        return r;
    }


    DType IQHVSolver::initAndSolveIQHV(Point& idealPoint, Point& nadirPoint, int numberOfSolutions,  bool callbacks)
    {
        int reserve_size = 4 * numberOfSolutions * pow(2, currentSettings->NumberOfObjectives / 2);
        //int reserve_size = 20000000;
        //int reserve_size = 5000;
        backend::ExecutionService* poolService = &(backend::ExecutionService::getInstance());
        backend::ExecutionPool* pool = &(poolService->getPool());
        int contextId = pool->reserveContext(reserve_size, currentSettings->nPoints, currentSettings->NumberOfObjectives,backend::ExecutionContext::ExecutionContextType::IQHVContext);
        backend::IQHVExecutionContext* context = (backend::IQHVExecutionContext*)pool->getContext(contextId);
        unsigned int i; for (i = 0; i < numberOfSolutions; i++) {
            if ((currentlySolvedProblem->points)[i] == NULL) {
                continue;
            }
            (*context->points)[i] = new Point(*currentlySolvedProblem->points[i]);

        }
        maxIndexUsed = numberOfSolutions - 1;
        context->maxIndexUsed = maxIndexUsed;
        context->maxIndexUsedOverall = maxIndexUsed;
        double result = backend::IQHV(0, numberOfSolutions - 1, contextId, idealPoint, nadirPoint, 0, currentSettings->NumberOfObjectives, 0, numberOfSolutions);
        //std::cout << currentSettings->name << "\t" << currentSettings->NumberOfObjectives << "\t" << currentSettings->nPoints << "\t" <<memoryManager->maxIndexUsedNumbers[contextId]  << "\t" << memoryManager->maxMaxIndexUsedNumbers[contextId];
        pool->releaseContext(contextId);
        pool->cleanMemory();
        return result;
        

    }

}
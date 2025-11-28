#include "DBHVESolver.h"
#include <random>
#include "NDTree.h"
#include "MemoryManager.h"
namespace moda {
    DBHVEResult* DBHVESolver::Solve(DataSet* problem, DBHVEParameters settings)
    {
        //initialize the problem
        prepareData(problem, settings);

        //call the starting callback
        std::string name = "HVE";
        StartCallback(*currentSettings, name);
        DBHVEResult* r = new DBHVEResult();

        r->type = Result::ResultType::Estimation;
        //calculate the hypervolume
        it0 = clock();
        r->HyperVolumeEstimation = initAndSolveHVE(currentlySolvedProblem->points, *betterPoint, *worsePoint, currentSettings->nPoints, settings.MCiterations, settings.MaxEstimationTime, settings.seed, settings.callbacks);
        r->ElapsedTime = clock() - it0;
        r->FinalResult = true;
        //call the closing callback
        EndCallback(*currentSettings, r);
        delete currentlySolvedProblem;
        //return the result
        return r;
    }

    DType DBHVESolver::initAndSolveHVE(std::vector <Point*>& set, Point& idealPoint, Point& nadirPoint, int numberOfSolutions, int MCiterations, clock_t runtimeLimit,
        unsigned seed,  bool callbacks)
    {
        Point referencePoint;

        int reserve_size = 4 * numberOfSolutions * pow(2, currentSettings->NumberOfObjectives / 2);
        //int reserve_size = 20000000;
        //int reserve_size = 5000;
        auto memoryManager = &(backend::ContextPool::getInstance());
        int memorySlot = memoryManager->reservePoints(reserve_size, currentSettings->nPoints, currentSettings->NumberOfObjectives);
        unsigned int i; for (i = 0; i < numberOfSolutions; i++) {
            if ((currentlySolvedProblem->points)[i] == NULL) {
                continue;
            }
            (*memoryManager->PointsTable[memorySlot])[i] = new Point(*currentlySolvedProblem->points[i]);

        }

        maxIndexUsed = numberOfSolutions - 1;
        memoryManager->maxIndexUsedNumbers[memorySlot] = maxIndexUsed;
        memoryManager->maxMaxIndexUsedNumbers[memorySlot] = maxIndexUsed;

        DType result = backend::DBHVE(memorySlot, idealPoint, nadirPoint, numberOfSolutions, MCiterations, runtimeLimit, seed, currentSettings->NumberOfObjectives, referencePoint, it0);
        backend::ContextPool::getInstance().releaseMemory(memorySlot);
        return result;
    }


  
    
}
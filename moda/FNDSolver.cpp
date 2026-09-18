#include "FNDSolver.h"
#include <algorithm>
#include "Hypervolume.h"
#include "FilterNonDominated.h"
#include "ExecutionContext.h"
#include "ExecutionPool.h"
#include "ExecutionService.h"
namespace moda {
	FNDResult* FNDSolver::Solve(DataSet* problem, FNDParameters parameters)
    {


        //initialize the poblem
		prepareData(problem, parameters);
		//call the starting callback
		std::string solverType = "";
		if (parameters.decomposition == FNDParameters::Decomposition::IQHV)  solverType = "IQHV Decomposition";
		else if (parameters.decomposition == FNDParameters::Decomposition::QHV)
		solverType = "QHV Decomposition";

		StartCallback(*currentSettings, "Filter Non Dominated  Contribution Solver type " + solverType);
		
        //initialize an empty result



		it0 = clock();

		FNDResult* r = new FNDResult(solveFND(currentlySolvedProblem->points, currentSettings->nPoints, worsePoint, betterPoint));

		r->type = Result::ResultType::SubsetSelection;
        r->ElapsedTime = clock() - it0;
        r->FinalResult = true;
        //call the closing callback
        EndCallback(*currentSettings, r);
		delete currentlySolvedProblem;
        //return the result
        return r;
    }




	FNDResult FNDSolver::solveFND(std::vector <Point*>& set, int numberOfSolutions, Point* worse, Point* better)
	{
		//int reserve_size = 4*numberOfSolutions * numberOfSolutions * currentSettings->NumberOfObjectives;
		//int reserve_size = 20000000;
		int reserve_size = 5000;
		backend::ExecutionService* poolService = &(backend::ExecutionService::getInstance());
		backend::ExecutionPool* pool = &(poolService->getPool());
		int contextId = pool->reserveContext(reserve_size, currentSettings->nPoints, currentSettings->NumberOfObjectives, backend::ExecutionContext::ExecutionContextType::FDPContext, false);
		backend::FDPExecutionContext* context = (backend::FDPExecutionContext*)pool->getContext(contextId);
		std::vector potentiallyDominated = std::vector<int>();
		std::vector insideBox = std::vector<int>();
		unsigned int i; for (i = 0; i < numberOfSolutions; i++) {
			if ((currentlySolvedProblem->points)[i] == NULL) {
				continue;
			}
			(*context->points)[i] = new Point(*currentlySolvedProblem->points[i]);
			insideBox.push_back(i);

		}
		context->maxIndexUsed = numberOfSolutions - 1;
		FNDResult result = backend::FilterNonDominated(insideBox, potentiallyDominated, contextId, *better, *worse,0,currentSettings->NumberOfObjectives,true,it0,IterationCallback);
		pool->releaseContext(contextId);
		return result;
	};
}
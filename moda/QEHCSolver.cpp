#include "QEHCSolver.h"
#include <algorithm>
#include "Hypervolume.h"
#include "QEHC.h"
#include "QEHC_parallel.h"
#include "QEHC_parallel_take2.h"
#include "ExecutionContext.h"
#include "ExecutionPool.h"
#include "ExecutionService.h"
namespace moda {
	QEHCResult* QEHCSolver::Solve(DataSet* problem, QEHCParameters parameters)
    {


        //initialize the poblem
		prepareData(problem, parameters);
		//call the starting callback
		std::string solverType = "";
		if (parameters.sort) solverType = "QEHC-sort";
		if (!parameters.sort && parameters.shuffle) solverType = "QEHC-randomShuffle";
		if (!parameters.sort && !parameters.shuffle) solverType = "QEHC-offset";
		//if ( parameters.sort && parameters.iterationsLimit == ULONG_MAX) solverType = "REF/REF*()";
		StartCallback(*currentSettings, "QEHC Contribution Solver type " + solverType);
		
        //initialize an empty result
        QEHCResult* r = new QEHCResult();
		QEHCResult r_prim;

		it0 = clock();

		r_prim = (solveQEHC(currentlySolvedProblem->points, currentSettings->nPoints, parameters.SearchSubject, worsePoint, parameters.sort, parameters.shuffle, parameters.offset, parameters.maxlevel, parameters.iterationsLimit));
		r->MinimumContribution = r_prim.MinimumContribution;
		r->MaximumContribution = r_prim.MaximumContribution;
		r->MinimumContributionIndex = r_prim.MinimumContributionIndex;
		r->MaximumContributionIndex = r_prim.MaximumContributionIndex;
		r->type = Result::ResultType::Contribution;
        r->ElapsedTime = clock() - it0;
        r->FinalResult = true;
        //call the closing callback
        EndCallback(*currentSettings, r);
		delete currentlySolvedProblem;
        //return the result
        return r;
    }




	QEHCResult QEHCSolver::solveQEHC(std::vector <Point*>& set, int numberOfSolutions, QEHCParameters::SearchSubjectOption searchSubject, Point* nadir, bool useSort, bool useShuffle, int offset, int maxlevel, unsigned long int iterationLimit)
	{
		//int reserve_size = 4*numberOfSolutions * numberOfSolutions * currentSettings->NumberOfObjectives;
		//int reserve_size = 20000000;
		int reserve_size = 5000;
		backend::ExecutionService* poolService = &(backend::ExecutionService::getInstance());
		backend::ExecutionPool* pool = &(poolService->getPool());
		int contextId = pool->reserveContext(reserve_size, currentSettings->nPoints, currentSettings->NumberOfObjectives, backend::ExecutionContext::ExecutionContextType::QEHCContext, false);
		backend::QEHCExecutionContext* context = (backend::QEHCExecutionContext*)pool->getContext(contextId);
		unsigned int i; for (i = 0; i < numberOfSolutions; i++) {
			if ((currentlySolvedProblem->points)[i] == NULL) {
				continue;
			}
			(*context->points)[i] = new Point(*currentlySolvedProblem->points[i]);

		}
		context->maxIndexUsed = numberOfSolutions - 1;
		QEHCResult result = backend::QEHC(contextId,numberOfSolutions, maxlevel,searchSubject,useSort,useShuffle,offset,iterationLimit,currentSettings->NumberOfObjectives, *nadir);
		pool->releaseContext(contextId);
		return result;
	};
}
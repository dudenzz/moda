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
	QEHCResult* QEHCSolver::Solve(DataSet* problem, QEHCParameters settings)
    {


        //initialize the poblem
		prepareData(problem, settings);
		//call the starting callback
		std::string solverType = "";
		if (settings.sort) solverType = "QEHC-sort";
		if (!settings.sort && settings.shuffle) solverType = "QEHC-randomShuffle";
		if (!settings.sort && !settings.shuffle) solverType = "QEHC-offset";
		//if ( parameters.sort && parameters.iterationsLimit == ULONG_MAX) solverType = "REF/REF*()";
		StartCallback(*currentSettings, "QEHC Contribution Solver type " + solverType);
		
        //initialize an empty result
        QEHCResult* r = new QEHCResult();
		QEHCResult r_prim;

		it0 = clock();

		r_prim = (solveQEHC(currentlySolvedProblem->points, currentSettings->nPoints, settings.SearchSubject, settings.sort, settings.shuffle,settings.offset,settings.maxlevel, settings.iterationsLimit));
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




	QEHCResult QEHCSolver::solveQEHC(std::vector <Point*>& set, int numberOfSolutions, QEHCParameters::SearchSubjectOption searchSubject, bool useSort, bool useShuffle, int offset, int maxlevel, unsigned long int iterationLimit)
	{
		//int reserve_size = 4*numberOfSolutions * numberOfSolutions * currentSettings->NumberOfObjectives;
		int reserve_size = 20000000;
		//int reserve_size = 5000;
		backend::ExecutionService* poolService = &(backend::ExecutionService::getInstance());
		backend::ExecutionPool* pool = &(poolService->getPool());
		int contextId = pool->reserveContext(reserve_size, currentSettings->nPoints, currentSettings->NumberOfObjectives, backend::ExecutionContext::ExecutionContextType::QEHCContext);
		backend::QEHCExecutionContext* context = (backend::QEHCExecutionContext*)pool->getContext(contextId);
		unsigned int i; for (i = 0; i < numberOfSolutions; i++) {
			if ((currentlySolvedProblem->points)[i] == NULL) {
				continue;
			}
			(*context->points)[i] = new Point(*currentlySolvedProblem->points[i]);

		}
		context->maxIndexUsed = numberOfSolutions - 1;
		QEHCResult result = backend::QEHC(contextId,numberOfSolutions, maxlevel,searchSubject,useSort,useShuffle,offset,iterationLimit,currentSettings->NumberOfObjectives);
		pool->releaseContext(contextId);
		return result;
	};
}
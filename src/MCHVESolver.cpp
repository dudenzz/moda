#include "MCHVESolver.h"
#include "MCHV.h"
namespace moda {


	MCHVResult* MCHVESolver::Solve(DataSet* problem, MCHVParameters settings)
	{
		
		
		//initialize the poblem
		prepareData(problem,settings);
		StartCallback(*currentSettings, "MC Solver");
		
		MCHVResult* r = new MCHVResult();
		//specific to this solver callback parameter
		//callbackViewer = 0;
		//initialize an empty result
		std::tuple<DType, DType, DType> result;
		r->type = Result::ResultType::Estimation;
		//calculate the hypervolume
		std::vector<MCHVResult*> results;
		it0 = clock();

		result = backend::solveMCHV(currentlySolvedProblem->points,  *betterPoint,*worsePoint, settings.MaxEstimationTime, results, currentSettings->NumberOfObjectives);
		r->HyperVolumeEstimation = std::get<0>(result);
		r->LowerBound = std::get<1>(result);
		r->UpperBound = std::get<2>(result);
		r->ElapsedTime = clock() - it0;
		r->FinalResult = true;
		//call the closing callback
		EndCallback(*currentSettings, r);
		delete currentlySolvedProblem;
		//return the result
		return r;
	}

	
}
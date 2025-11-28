#include "QHV_BQSolver.h"
#include "QHV_BQ.h"
#include "Hypervolume.h"
namespace moda {
	


	QHV_BQResult* QHV_BQSolver::Solve(DataSet* problem, QHV_BQParameters settings)
	{

		//initialize the problem
		prepareData(problem, settings);

		indexSet.reserve(2000000000);
		indexSet.resize(20000000);
		subProblems = new SubproblemsPool<SubProblem>();

		//call the starting callback
		if (settings.MonteCarlo)
			StartCallback(*currentSettings, "QHV-BQ + MC Solver");
		else
			StartCallback(*currentSettings, "QHV-BQ Solver");
		
		//initialize an empty result
		QHV_BQResult* r = new QHV_BQResult();
		std::tuple<DType, DType, DType> result;
		r->type = Result::ResultType::Estimation;

		//calculate the hypervolume
		std::vector<QHV_BQResult*> results;
		it0 = clock();
		callbackViewer = 0;
		result = solveQHV_BQ(currentlySolvedProblem->points, *betterPoint, *worsePoint, currentSettings->nPoints, settings.MaxEstimationTime, settings.MonteCarlo, results, settings.SwitchToMCSettings);
		r->HyperVolumeEstimation = get<1>(result);
		r->LowerBound = get<0>(result);
		r->UpperBound = get<2>(result);
		r->ElapsedTime = clock() - it0;
		r->FinalResult = true;
		//call the closing callback
		EndCallback(*currentSettings, r);
		delete currentlySolvedProblem;
		delete subProblems;
		//return the result
		return r;
	}

	inline int QHV_BQSolver::off(int offset, int j) {
		return (j + offset) % currentSettings->NumberOfObjectives;
	}
	std::tuple<DType, DType, DType>  QHV_BQSolver::solveQHV_BQ(std::vector <Point*>& set, Point& idealPoint, Point& nadirPoint, int numberOfSolutions, clock_t maxTime, bool switch2MC,
		std::vector < QHV_BQResult* >& results, SwitchParameters mcSettings)
	{
		return backend::QHV_BQ(set, idealPoint, nadirPoint, numberOfSolutions, maxTime, switch2MC, results, mcSettings, currentSettings->NumberOfObjectives);
	}
	

	

}
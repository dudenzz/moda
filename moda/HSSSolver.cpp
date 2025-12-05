#include "HSSSolver.h"
#include "HSS.h"
#include "deque"
#include "map"
namespace moda {
	HSSResult* HSSSolver::Solve(DataSet* problem, HSSParameters settings)
    {
		//initialize the problem
		prepareData(problem, settings);
		//call the starting callback
		if (settings.Strategy)
			StartCallback(*currentSettings, "QHV-II based Incremental Subset Selection Solver");
		else
			StartCallback(*currentSettings, settings.Experimental ? "QHV-II based Decremental Subset Selection Solver (experimental)" : "QHV-II based Decremental Subset Selection Solver");




		//initialize an empty result
		HSSResult* r = new HSSResult();

		r->methodName = settings.Experimental ? "HSS experimental" : "HSS";
        r->type = Result::ResultType::SubsetSelection;
        //calculate the hypervolume
		std::vector<int> selectedPoints;
		it0 = clock();
        if(settings.Strategy == HSSParameters::SubsetSelectionStrategy::Incremental)
            r->HyperVolume = backend::greedyHSSIncLazyIQHV(currentlySolvedProblem->points, selectedPoints, *betterPoint, *worsePoint, settings.StoppingCriteria, settings.StoppingSubsetSize, settings.StoppingTime, settings.callbacks, settings.CalculateHV, currentSettings->NumberOfObjectives)->HyperVolume;
		else
		{
			r->HyperVolume = backend::greedyHSSDecLazyIQHV(currentlySolvedProblem->points, selectedPoints, *betterPoint, *worsePoint, settings.StoppingCriteria, settings.StoppingSubsetSize, settings.StoppingTime, settings.callbacks, settings.CalculateHV, currentSettings->NumberOfObjectives)->HyperVolume;
		}
		r->ElapsedTime = clock() - it0;
		r->selectedPoints = selectedPoints;
        r->FinalResult = true;
        //call the closing callback
        EndCallback(*currentSettings, r);
		delete currentlySolvedProblem;
        //return the result
        return r;
    }

	
	
};

    

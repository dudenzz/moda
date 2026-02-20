#include "QHV_BR.h"
#include "Hypervolume.h"
namespace moda {

	QHV_BRResult* QHV_BR::Solve(DataSet* problem, QHV_BRParameters settings)
	{
		//initialize the problem
		prepareData(problem, settings);
		//call the starting callback
		this->StartCallback(*currentSettings, "QHV-BR Solver");
		this->currentlySolvedProblem = new DataSet(*problem);
		
		//initialize an empty result
		std::tuple<DType, DType, DType> result;
		QHV_BRResult* r = new QHV_BRResult();

		r->type = Result::ResultType::Estimation;
		//calculate the hypervolume
		std::vector<Result*> results;
		callbackViewer = 0;

		result = initAndSolveBRHV(currentlySolvedProblem->points, *betterPoint, *worsePoint, currentSettings->nPoints, settings.MaxEstimationTime, results, settings.callbacks);
		r->HyperVolumeEstimation = std::get<1>(result);
		r->LowerBound = std::get<0>(result);
		r->UpperBound = std::get<2>(result);
		r->ElapsedTime = clock() - this->it0;
		r->FinalResult = true;
		//call the closing callback
		this->EndCallback(*currentSettings, r);
		delete currentlySolvedProblem;
		//return the result
		return r;
	}


	// Entry point to QHV with bounds calculation using recursion
	std::tuple<DType, DType, DType> QHV_BR::initAndSolveBRHV(std::vector <Point*>& set, Point& IdealPoint, Point& NadirPoint, int numberOfSolutions, clock_t maxTime,
		std::vector <Result*>& results, bool callbacks) {
		this->it0 = clock();
		this->currentlySolvedProblem->points.resize(20000000);
		this->point2 = new Point(currentSettings->NumberOfObjectives);
		unsigned int i; for (i = 0; i < numberOfSolutions; i++) {
			if ((set)[i] == NULL) {
				continue;
			}
			currentlySolvedProblem->points[i] = set[i];
		}

		Point newIdealPoint = Point(currentSettings->NumberOfObjectives);
		Point newNadirPoint = Point(currentSettings->NumberOfObjectives);
		int j;
		for (j = 0; j < currentSettings->NumberOfObjectives; j++) {
			newIdealPoint.ObjectiveValues[j] = 1; //@TODO hard coded definition of nadir
			newNadirPoint.ObjectiveValues[j] = 0; //@TODO hard coded definition of nadir
		}

		this->maxIndexUsed = numberOfSolutions - 1;

		DType subProblemVolume = backend::Backend::Hypervolume(&NadirPoint, &IdealPoint, &IdealPoint, currentSettings->NumberOfObjectives);
		lowerBoundVolume = 0;
		upperBoundVolume = subProblemVolume;

		clock_t t0 = clock();
		clock_t time = t0;
		DType totalVolume;
		if(callbacks)  totalVolume = SolveBRHV_callbacks(0, numberOfSolutions - 1, newIdealPoint, newNadirPoint, 0, maxTime, t0, time,
			results, 0,0);
		else  totalVolume = SolveBRHV(0, numberOfSolutions - 1, newIdealPoint, newNadirPoint, 0, maxTime, t0, time,
			results, 0, 0);
		std::tuple<DType, DType, DType> result(totalVolume, lowerBoundVolume, upperBoundVolume);
		return result;
	}
	inline int QHV_BR::off(int offset, int j) {
		return (j + offset) % currentSettings->NumberOfObjectives;
	}
	DType QHV_BR::SolveBRHV_callbacks(int start, int end, Point& IdealPoint, Point& NadirPoint, int offset,
		clock_t maxTime, clock_t& t0, clock_t& time, std::vector <Result*>& results, int recursion, int outerIteratorValue) {

		time = clock();

		BoundedResult* result = new BoundedResult;
		result->ElapsedTime = time - t0;
		result->HyperVolumeEstimation = (lowerBoundVolume + upperBoundVolume) / 2;
		result->LowerBound = lowerBoundVolume;
		result->UpperBound = upperBoundVolume;
		results.push_back(result);
		result->type = Result::Estimation;
		IterationCallback(callbackViewer, 0, result);
		callbackViewer += 1;

		

		if (end < start || clock() - t0 >= maxTime) {
			return 0;
		}

		DType thisVolume = Hypervolume(&NadirPoint, &IdealPoint);

		this->branches++;
		offset++;
		offset = offset % currentSettings->NumberOfObjectives;

		int oldmaxIndexUsed = this->maxIndexUsed;

		// If there is just one point
		if (end == start) {
			DType totalVolume = backend::Backend::Hypervolume(&NadirPoint, currentlySolvedProblem->points[start], &IdealPoint, currentSettings->NumberOfObjectives);

			lowerBoundVolume += totalVolume;
			upperBoundVolume -= thisVolume - totalVolume;

			return totalVolume;
		}

		// If there are just two points
		if (end - start == 1) {
			DType totalVolume = backend::Backend::Hypervolume(&NadirPoint, currentlySolvedProblem->points[start], &IdealPoint, currentSettings->NumberOfObjectives);
			totalVolume += backend::Backend::Hypervolume(&NadirPoint, currentlySolvedProblem->points[end], &IdealPoint, currentSettings->NumberOfObjectives);
			*this->point2 = *(currentlySolvedProblem->points[start]);
			unsigned j;
			for (j = 0; j < currentSettings->NumberOfObjectives; j++) {
				this->point2->ObjectiveValues[j] = std::min(this->point2->ObjectiveValues[j], currentlySolvedProblem->points[end]->ObjectiveValues[j]);
			}
			totalVolume -= backend::Backend::Hypervolume(&NadirPoint, point2, &IdealPoint, currentSettings->NumberOfObjectives);

			lowerBoundVolume += totalVolume;
			upperBoundVolume -= thisVolume - totalVolume;

			return totalVolume;
		}

		int iPivot = start;
		DType maxVolume;
		maxVolume = backend::Backend::Hypervolume(&NadirPoint, (currentlySolvedProblem->points)[iPivot], &IdealPoint, currentSettings->NumberOfObjectives);

		// Find the pivot point
		unsigned i;
		for (i = start + 1; i <= end; i++) {
			DType volumeCurrent;
			volumeCurrent = backend::Backend::Hypervolume(&NadirPoint, (currentlySolvedProblem->points)[i], &IdealPoint, currentSettings->NumberOfObjectives);
			if (maxVolume < volumeCurrent) {
				maxVolume = volumeCurrent;
				iPivot = i;
			}
		}

		DType totalVolume = backend::Backend::Hypervolume(&NadirPoint, (currentlySolvedProblem->points)[iPivot], &IdealPoint, currentSettings->NumberOfObjectives);

		lowerBoundVolume += totalVolume;

		// Build subproblems
		int iPos = this->maxIndexUsed + 1;
		unsigned j;
		int jj;

		int ic = 0;

		Point partNadirPoint = NadirPoint;
		Point partIdealPoint = IdealPoint;

		for (jj = 0; jj < currentSettings->NumberOfObjectives; jj++) {
			j = off(offset, jj);

			if (jj > 0) {
				int j2 = off(offset, jj - 1);
				partIdealPoint.ObjectiveValues[j2] = std::min(IdealPoint.ObjectiveValues[j2], currentlySolvedProblem->points[iPivot]->ObjectiveValues[j2]);
				partNadirPoint.ObjectiveValues[j2] = NadirPoint.ObjectiveValues[j2];
			}

			int partStart = iPos;

			for (i = start; i <= end; i++) {
				if (i == iPivot)
					continue;

				if (std::min(IdealPoint.ObjectiveValues[j], currentlySolvedProblem->points[i]->ObjectiveValues[j]) >
					std::min(IdealPoint.ObjectiveValues[j], (currentlySolvedProblem->points)[iPivot]->ObjectiveValues[j])) {
					currentlySolvedProblem->points[iPos++] = currentlySolvedProblem->points[i];
				}

			}
			int partEnd = iPos - 1;

			this->maxIndexUsed = iPos - 1;

			partNadirPoint.ObjectiveValues[j] = std::min(IdealPoint.ObjectiveValues[j], currentlySolvedProblem->points[iPivot]->ObjectiveValues[j]);

			if (partEnd >= partStart) {

				totalVolume += SolveBRHV(partStart, partEnd, partIdealPoint, partNadirPoint, offset, maxTime, t0, time,
					results, recursion + 1, jj);
			}
			else {
				upperBoundVolume -= this->Hypervolume(&partIdealPoint, &partNadirPoint);
			}
		}

		this->maxIndexUsed = oldmaxIndexUsed;
		return totalVolume;
	}
	DType QHV_BR::SolveBRHV(int start, int end, Point& IdealPoint, Point& NadirPoint, int offset,
		clock_t maxTime, clock_t& t0, clock_t& time, std::vector <Result*>& results, int recursion, int outerIteratorValue) {

		time = clock();

		BoundedResult* result = new BoundedResult;
		result->ElapsedTime = time - t0;
		result->HyperVolumeEstimation = (lowerBoundVolume + upperBoundVolume)/2;
		result->LowerBound = lowerBoundVolume;
		result->UpperBound = upperBoundVolume;
		results.push_back(result);
		result->type = Result::Estimation;


		

		if (end < start || clock() - t0 >= maxTime) {
			return 0;
		}

		DType thisVolume = this->Hypervolume(&NadirPoint, &IdealPoint);

		this->branches++;
		offset++;
		offset = offset % currentSettings->NumberOfObjectives;

		int oldmaxIndexUsed = this->maxIndexUsed;

		// If there is just one point
		if (end == start) {
			DType totalVolume = backend::Backend::Hypervolume(&NadirPoint, currentlySolvedProblem->points[start], &IdealPoint, currentSettings->NumberOfObjectives);

			lowerBoundVolume += totalVolume;
			upperBoundVolume -= thisVolume - totalVolume;

			return totalVolume;
		}

		// If there are just two points
		if (end - start == 1) {
			DType totalVolume = backend::Backend::Hypervolume(&NadirPoint, currentlySolvedProblem->points[start], &IdealPoint, currentSettings->NumberOfObjectives);
			totalVolume += backend::Backend::Hypervolume(&NadirPoint, currentlySolvedProblem->points[end], &IdealPoint, currentSettings->NumberOfObjectives);
			*this->point2 = *(currentlySolvedProblem->points[start]);
			unsigned j;
			for (j = 0; j < currentSettings->NumberOfObjectives; j++) {
				this->point2->ObjectiveValues[j] = std::min(this->point2->ObjectiveValues[j], currentlySolvedProblem->points[end]->ObjectiveValues[j]);
			}
			totalVolume -= backend::Backend::Hypervolume(&NadirPoint, point2, &IdealPoint, currentSettings->NumberOfObjectives);

			lowerBoundVolume += totalVolume;
			upperBoundVolume -= thisVolume - totalVolume;

			return totalVolume;
		}

		int iPivot = start;
		DType maxVolume;
		maxVolume = backend::Backend::Hypervolume(&NadirPoint, (currentlySolvedProblem->points)[iPivot], &IdealPoint, currentSettings->NumberOfObjectives);

		// Find the pivot point
		unsigned i;
		for (i = start + 1; i <= end; i++) {
			DType volumeCurrent;
			volumeCurrent = backend::Backend::Hypervolume(&NadirPoint, (currentlySolvedProblem->points)[i], &IdealPoint, currentSettings->NumberOfObjectives);
			if (maxVolume < volumeCurrent) {
				maxVolume = volumeCurrent;
				iPivot = i;
			}
		}

		DType totalVolume = backend::Backend::Hypervolume(&NadirPoint, (currentlySolvedProblem->points)[iPivot], &IdealPoint, currentSettings->NumberOfObjectives);

		lowerBoundVolume += totalVolume;

		// Build subproblems
		int iPos = this->maxIndexUsed + 1;
		unsigned j;
		int jj;

		int ic = 0;

		Point partNadirPoint = NadirPoint;
		Point partIdealPoint = IdealPoint;

		for (jj = 0; jj < currentSettings->NumberOfObjectives; jj++) {
			j = off(offset, jj);

			if (jj > 0) {
				int j2 = off(offset, jj - 1);
				partIdealPoint.ObjectiveValues[j2] = std::min(IdealPoint.ObjectiveValues[j2], currentlySolvedProblem->points[iPivot]->ObjectiveValues[j2]);
				partNadirPoint.ObjectiveValues[j2] = NadirPoint.ObjectiveValues[j2];
			}

			int partStart = iPos;

			for (i = start; i <= end; i++) {
				if (i == iPivot)
					continue;

				if (std::min(IdealPoint.ObjectiveValues[j], currentlySolvedProblem->points[i]->ObjectiveValues[j]) >
					std::min(IdealPoint.ObjectiveValues[j], (currentlySolvedProblem->points)[iPivot]->ObjectiveValues[j])) {
					currentlySolvedProblem->points[iPos++] = currentlySolvedProblem->points[i];
				}

			}
			int partEnd = iPos - 1;

			this->maxIndexUsed = iPos - 1;

			partNadirPoint.ObjectiveValues[j] = std::min(IdealPoint.ObjectiveValues[j], currentlySolvedProblem->points[iPivot]->ObjectiveValues[j]);

			if (partEnd >= partStart) {

				totalVolume += SolveBRHV(partStart, partEnd, partIdealPoint, partNadirPoint, offset, maxTime, t0, time,
					results, recursion + 1, jj);
			}
			else {
				upperBoundVolume -= this->Hypervolume(&partIdealPoint, &partNadirPoint);
			}
		}

		this->maxIndexUsed = oldmaxIndexUsed;
		return totalVolume;
	}

}
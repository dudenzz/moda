#include "IQHV_experimental2.h"
namespace qhv {
    Result* IQHV_experimental2::Solve(DataSet* problem, SolverSettings settings)
    {
        //initialize the poblem
        this->currentlySolvedProblem = new DataSet(*problem);
        this->currentSettings = problem->getParameters();
        this->currentlySolvedProblem->setParameters(this->currentSettings);
        //deep copy the problem data
        this->indexSet = new Point * [this->currentSettings.nPoints];
        copy(problem->points.begin(), problem->points.end(), this->indexSet);
        if (settings.minimize && problem->typeOfOptimization == DataSet::OptimizationType::maximalization)
        {
            currentlySolvedProblem->reverseObjectives();
            problem->typeOfOptimization = DataSet::OptimizationType::minimalization;
        }
        if (!settings.minimize && problem->typeOfOptimization == DataSet::OptimizationType::minimalization)
        {
            currentlySolvedProblem->reverseObjectives();
            problem->typeOfOptimization = DataSet::OptimizationType::maximalization;
        }
        //initialize solution variables
        this->branches = 0;
        this->leafs = 0;
        this->maxIndexUsed = 0;

        //call the starting callback
        StartCallback(this->currentSettings, "IQHV Solver");

        // normalize
        if (this->currentlySolvedProblem->getParameters().normalize)
        {
            problem->normalize();
        }
        //get or calculate the better and worse point
        Point betterPoint = *settings.betterReferencePoint;
        Point worsePoint = *settings.worseReferencePoint;

        //initialize an empty result
        HypervolumeResult* r = new HypervolumeResult();

        r->type = Result::ResultType::Hypervolume;
        //calculate the hypervolume
        r->Volume = initAndSolveIQHV(problem->points, betterPoint, worsePoint, this->currentSettings.nPoints, false, false);
        r->ElapsedTime = clock() - this->it0;
        r->FinalResult = true;
        //call the closing callback
        this->EndCallback(this->currentSettings, r);

        //return the result
        return r;
    }

    Result* IQHV_experimental2::Solve(DataSet* problem)
    {
        //initialize the poblem
        this->currentlySolvedProblem = new DataSet(*problem);
        this->currentSettings = problem->getParameters();
        this->currentlySolvedProblem->setParameters(this->currentSettings);
        //deep copy the problem data
        this->indexSet = new Point * [this->currentSettings.nPoints];
        copy(problem->points.begin(), problem->points.end(), this->indexSet);

        //initialize solution variables
        this->branches = 0;
        this->leafs = 0;
        this->maxIndexUsed = 0;

        //call the starting callback
        StartCallback(this->currentSettings, "IQHV Solver");

        //get or calculate the better and worse point
        Point betterPoint = Point::ones(this->currentSettings.NumberOfObjectives);
        //Point betterPoint = *currentlySolvedProblem->getBetter();
        Point worsePoint = Point::zeroes(this->currentSettings.NumberOfObjectives);
        //Point worsePoint = *currentlySolvedProblem->getWorse();
        // cout << betterPoint << endl << worsePoint;
        // normalize
        if (this->currentlySolvedProblem->getParameters().normalize) problem->normalize();
        //initialize an empty result
        HypervolumeResult *r = new HypervolumeResult();

        //calculate the hypervolume
        r->Volume = initAndSolveIQHV(problem->points, betterPoint, worsePoint, this->currentSettings.nPoints, false, false);
        r->ElapsedTime = clock() - this->it0;
        r->FinalResult = true;
        //call the closing callback
        this->EndCallback(this->currentSettings, r);

        //return the result
        return r;
    }

    //int start, int end, TPoint& approximateIdealPoint, TPoint& approximateNadirPoint, int offset, bool bRemoveDominated, bool cut/*, int depth*/
    DType IQHV_experimental2::initAndSolveIQHV(vector<Point*>& set, Point& idealPoint, Point& nadirPoint, int numberOfSolutions,
        bool bRemoveDominated, bool cut)
    {
        this->it0 = clock();
        point2 = new Point;
        if (this->indexSet != NULL)
            delete this->indexSet;

        this->indexSet = new Point * [set.size() * 10000];
        unsigned int i; for (i = 0; i < numberOfSolutions; i++) {
            if ((set)[i] == NULL) {
                continue;
            }
            this->indexSet[i] = set[i];
        }

        Point newIdealPoint = Point(this->currentSettings.NumberOfObjectives);
        Point newNadirPoint = Point(this->currentSettings.NumberOfObjectives);
        int j;
        for (j = 0; j < this->currentSettings.NumberOfObjectives; j++) {
            newIdealPoint.ObjectiveValues[j] = 1;
            newNadirPoint.ObjectiveValues[j] = 0;
        }

        this->maxIndexUsed = numberOfSolutions - 1;

        return solveIQHV(0, numberOfSolutions - 1, newIdealPoint, newNadirPoint, 0, bRemoveDominated, cut/*, 1*/, 0, 0);
    }

    // zadanie: porównanie metod z 2015 roku i 2024 - porównanie czasów wykonania.
    DType IQHV_experimental2::solveIQHV(int start, int end, Point& IdealPoint, Point& NadirPoint, int offset, bool bRemoveDominated, bool cut, int recursion, int outerIteratorValue)
    {

        //HypervolumeResult tempResult;
        if (bRemoveDominated) {
            this->RemoveDominated(start, end, IdealPoint);
        }

        if (end < start) {
            return 0;
        }

        this->branches++;

        int N = 10;
        if (this->currentSettings.NumberOfObjectives == 4)
            N = 8;
        if (this->currentSettings.NumberOfObjectives == 6)
            N = 9;

        if (cut && (end - start + 1 < N)) {
            this->leafs++;
            return 0;
        }

        offset++;
        offset = offset % this->currentSettings.NumberOfObjectives;

        int oldmaxIndexUsed = this->maxIndexUsed;

        // If there is just one point
        if (end == start) {
            DType totalVolume = this->Volume(&NadirPoint, (this->indexSet[start]), &IdealPoint);
            //		cout << "v= " << totalVolume << '\n';
            return totalVolume;
        }

        // If there are just two points
        if (end - start == 1) {
            DType totalVolume = this->Volume(&NadirPoint, (this->indexSet[start]), &IdealPoint);
            totalVolume += this->Volume(&NadirPoint, (this->indexSet[end]), &IdealPoint);
            *(point2) = *(this->indexSet[start]);
            unsigned j;
            for (j = 0; j < this->currentSettings.NumberOfObjectives; j++) {
                point2->ObjectiveValues[j] = min(this->point2->ObjectiveValues[j], this->indexSet[end]->ObjectiveValues[j]);
            }
            totalVolume -= this->Volume(&NadirPoint, point2, &IdealPoint);
            return totalVolume;
        }

        int iPivot = start;
        DType maxVolume;
        maxVolume = this->Volume(&NadirPoint, (this->indexSet)[iPivot], &IdealPoint);

        // Find the pivot point
        unsigned i;
        for (i = start + 1; i <= end; i++) {
            DType volumeCurrent;
            volumeCurrent = this->Volume(&NadirPoint, (this->indexSet)[i], &IdealPoint);
            if (maxVolume < volumeCurrent) {
                maxVolume = volumeCurrent;
                iPivot = i;
            }
        }

        DType totalVolume = this->Volume(&NadirPoint, (this->indexSet)[iPivot], &IdealPoint);
        //	cout << "pv= " << maxVolume << '\n';

        vector <int> starts, ends;
        starts.resize(this->currentSettings.NumberOfObjectives);
        ends.resize(this->currentSettings.NumberOfObjectives);

        // Build subproblems
        int iPos = this->maxIndexUsed + 1;
        unsigned j;
        unsigned jj;
        for (jj = 0; jj < this->currentSettings.NumberOfObjectives; jj++) {
            j = off(offset, jj);
            starts[j] = iPos;

            for (i = start; i <= end; i++) {
                if (i == iPivot)
                    continue;

                if (min(IdealPoint.ObjectiveValues[j], this->indexSet[i]->ObjectiveValues[j]) >
                    min(IdealPoint.ObjectiveValues[j], (this->indexSet)[iPivot]->ObjectiveValues[j])) {
                    this->indexSet[iPos++] = this->indexSet[i];
                }
            }

            ends[j] = iPos - 1;
        }

        this->maxIndexUsed = iPos - 1;

        //if (recursion == 0)
        //{
        //    tempResult.ElapsedTime = clock() - it0;
        //    tempResult.Volume = 0.0;
        //    tempResult.FinalResult = false;
        //    tempResult.type = Result::Hypervolume;
        //    IterationCallback(0, currentSettings.NumberOfObjectives * currentSettings.NumberOfObjectives, &tempResult);
        //}
        // Recursive call for each subproblem
        for (jj = 0; jj < this->currentSettings.NumberOfObjectives; jj++) {
            j = off(offset, jj);
            //		cout << ends[j] - starts[j] + 1 << "\n";
            if (ends[j] >= starts[j]) {
                Point partNadirPoint = NadirPoint;
                Point partIdealPoint = IdealPoint;
                partNadirPoint.ObjectiveValues[j] = min(IdealPoint.ObjectiveValues[j], this->indexSet[iPivot]->ObjectiveValues[j]);
                int j2;
                int jj2;
                for (jj2 = 0; jj2 < jj; jj2++) {
                    j2 = off(offset, jj2);
                    partIdealPoint.ObjectiveValues[j2] = min(IdealPoint.ObjectiveValues[j2], this->indexSet[iPivot]->ObjectiveValues[j2]);
                }
                totalVolume += solveIQHV(starts[j], ends[j], partIdealPoint, partNadirPoint, offset, bRemoveDominated, cut/*, depth + 1*/, recursion + 1, jj);
                //if (recursion == 1)
                //{

                //    tempResult.ElapsedTime = clock() - it0;
                //    tempResult.Volume = totalVolume;
                //    tempResult.type = Result::Hypervolume;
                //    IterationCallback(outerIteratorValue * currentSettings.NumberOfObjectives + jj + 1, currentSettings.NumberOfObjectives * currentSettings.NumberOfObjectives, &tempResult);
                //}
            }
        }
        this->maxIndexUsed = oldmaxIndexUsed;
        //	cout << totalVolume << '\n';
        return (totalVolume);


    }
}
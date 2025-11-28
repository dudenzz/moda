#include "IQHV_Contribution.h"
namespace qhv {
    Result* IQHV_Contribution::Solve(DataSet* problem, SolverSettings settings)
    {
        //initialize the poblem

        currentlySolvedProblem = new DataSet(*problem);
        currentSettings = problem->getParameters();
        currentlySolvedProblem->setParameters(currentSettings);
        //deep copy the problem data
        indexSet = new Point * [currentSettings.nPoints];
        indexSetVec = problem->points;
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
        branches = 0;
        leafs = 0;
        maxIndexUsed = 0;

        //call the starting callback
        StartCallback(currentSettings, "IQHV Contribution Solver");

        // normalize
        if (currentlySolvedProblem->getParameters().normalize)
        {
            problem->normalize();
        }
        //get or calculate the better and worse point
        Point betterPoint = *settings.betterReferencePoint;
        Point worsePoint = *settings.worseReferencePoint;

        //initialize an empty result
        ContributionResult* r = new ContributionResult();

        r->type = Result::ResultType::Contribution;
        r->FinalResult = false;
        r->MaximumContribution = 0;
        r->MinimumContribution = 100000;
        r->MaximumContributionIndex = -1;
        r->MinimumContributionIndex = -1;
        DType contribution;
        //calculate the hypervolume


        it0 = clock();
        for (int i0 = 0; i0 < currentSettings.nPoints; i0++) {
            contribution = solveIQHV_contribution(i0, problem->points, betterPoint, worsePoint, currentSettings.nPoints, RUNTIME_10_MIN, settings.callbacks);
            if (r->MaximumContribution <= contribution) {
                r->MaximumContribution = contribution;
                r->MaximumContributionIndex = i0;
            }
            if (r->MinimumContribution >= contribution) {
                r->MinimumContribution = contribution;
                r->MinimumContributionIndex = i0;
            }
            if (settings.callbacks) {
                IterationCallback(i0, currentSettings.nPoints, r);
            }
        }

        r->ElapsedTime = clock() - it0;
        r->FinalResult = true;
        //call the closing callback
        EndCallback(currentSettings, r);

        //return the result
        return r;
    }
    Result* IQHV_Contribution::Solve(DataSet* problem)
    {
        throw std::exception("Not Implemented.");
        Result* r = new Result();
        return r;
    }

    DType IQHV_Contribution::solveIQHV_contribution(int newPoint, vector<Point*>& set, Point& idealPoint, Point& nadirPoint, int numberOfSolutions,
        clock_t runtimeLimit, bool callbacks)
    {
        indexSetVec.resize(20000000);
        point2 = new Point;
        int i2 = 0;
        unsigned int i; for (i = 0; i < numberOfSolutions; i++) {
            if (i != newPoint)
            {
                if ((set)[i] == NULL) {
                    continue;
                }
                indexSetVec[i2++] = set[i];
            }
        }

        Point newIdealPoint, newNadirPoint;
        int j;
        for (j = 0; j < currentSettings.NumberOfObjectives; j++) {
            newIdealPoint.ObjectiveValues[j] = set[newPoint]->ObjectiveValues[j];
            newNadirPoint.ObjectiveValues[j] = 0;
        }

        int start = 0;
        int end = numberOfSolutions - 2;

        maxIndexUsed = numberOfSolutions - 1;
        return Volume(&newNadirPoint, &newIdealPoint) - IQHV(start, end, newIdealPoint, newNadirPoint, 0);
    }



    DType IQHV_Contribution::IQHV(int start, int end, Point& IdealPoint, Point& NadirPoint, short offset)
    {
        clock_t t0 = clock();

        if (end < start) {
            return 0;
        }

        offset++;
        offset = offset % currentSettings.NumberOfObjectives;

        int oldmaxIndexUsed = maxIndexUsed;

        // If there is just one point
        if (end == start) {
            double totalVolume = Volume(&NadirPoint, (indexSetVec[start]), &IdealPoint);
            return totalVolume;
        }

        // If there are just two points
        if (end - start == 1) {
            double totalVolume = Volume(&NadirPoint, (indexSetVec[start]), &IdealPoint);
            totalVolume += Volume(&NadirPoint, (indexSetVec[end]), &IdealPoint);
            *point2 = *(indexSetVec[start]);
            unsigned j;
            for (j = 0; j < currentSettings.NumberOfObjectives; j++) {
                point2->ObjectiveValues[j] = min(point2->ObjectiveValues[j], indexSetVec[end]->ObjectiveValues[j]);
            }

            totalVolume -= Volume(&NadirPoint, point2, &IdealPoint);
            return totalVolume;
        }

        int iPivot = start;
        double maxVolume;
        maxVolume = Volume(&NadirPoint, (indexSetVec)[iPivot], &IdealPoint);

        // Find the pivot point
        unsigned i;
        for (i = start + 1; i <= end; i++) {
            double volumeCurrent;
            volumeCurrent = Volume(&NadirPoint, (indexSetVec)[i], &IdealPoint);
            if (maxVolume < volumeCurrent) {
                maxVolume = volumeCurrent;
                iPivot = i;
            }
        }

        double totalVolume = Volume(&NadirPoint, (indexSetVec)[iPivot], &IdealPoint);

        // Build subproblems
        int iPos = maxIndexUsed + 1;
        unsigned j;
        int jj;

        int ic = 0;

        Point partNadirPoint = NadirPoint;
        Point partIdealPoint = IdealPoint;

        for (jj = 0; jj < currentSettings.NumberOfObjectives; jj++) {
            j = off(offset, jj);

            if (jj > 0) {
                int j2 = off(offset, jj - 1);
                partIdealPoint.ObjectiveValues[j2] = min(IdealPoint.ObjectiveValues[j2], indexSetVec[iPivot]->ObjectiveValues[j2]);
                partNadirPoint.ObjectiveValues[j2] = NadirPoint.ObjectiveValues[j2];
            }

            int partStart = iPos;

            for (i = start; i <= end; i++) {
                if (i == iPivot)
                    continue;

                if (min(IdealPoint.ObjectiveValues[j], indexSetVec[i]->ObjectiveValues[j]) >
                    min(IdealPoint.ObjectiveValues[j], (indexSetVec)[iPivot]->ObjectiveValues[j])) {
                    indexSetVec[iPos++] = indexSetVec[i];
                }
            }

            int partEnd = iPos - 1;

            maxIndexUsed = iPos - 1;

            if (partEnd >= partStart) {
                partNadirPoint.ObjectiveValues[j] = min(IdealPoint.ObjectiveValues[j], indexSetVec[iPivot]->ObjectiveValues[j]);

                totalVolume += IQHV(partStart, partEnd, partIdealPoint, partNadirPoint, offset);
            }
        }

        maxIndexUsed = oldmaxIndexUsed;

        return (totalVolume);
    }
    DType IQHV_Contribution::IQHV_callbacks(int start, int end, Point& IdealPoint, Point& NadirPoint, short offset, int recursion, int outerIteratorValue)
    {
        clock_t t0 = clock();

        if (end < start) {
            return 0;
        }

        offset++;
        offset = offset % currentSettings.NumberOfObjectives;

        int oldmaxIndexUsed = maxIndexUsed;

        // If there is just one point
        if (end == start) {
            double totalVolume = Volume(&NadirPoint, (indexSetVec[start]), &IdealPoint);
            return totalVolume;
        }

        // If there are just two points
        if (end - start == 1) {
            double totalVolume = Volume(&NadirPoint, (indexSetVec[start]), &IdealPoint);
            totalVolume += Volume(&NadirPoint, (indexSetVec[end]), &IdealPoint);
            *point2 = *(indexSetVec[start]);
            unsigned j;
            for (j = 0; j < currentSettings.NumberOfObjectives; j++) {
                point2->ObjectiveValues[j] = min(point2->ObjectiveValues[j], indexSetVec[end]->ObjectiveValues[j]);
            }

            totalVolume -= Volume(&NadirPoint, point2, &IdealPoint);
            return totalVolume;
        }

        int iPivot = start;
        double maxVolume;
        maxVolume = Volume(&NadirPoint, (indexSetVec)[iPivot], &IdealPoint);

        // Find the pivot point
        unsigned i;
        for (i = start + 1; i <= end; i++) {
            double volumeCurrent;
            volumeCurrent = Volume(&NadirPoint, (indexSetVec)[i], &IdealPoint);
            if (maxVolume < volumeCurrent) {
                maxVolume = volumeCurrent;
                iPivot = i;
            }
        }

        double totalVolume = Volume(&NadirPoint, (indexSetVec)[iPivot], &IdealPoint);

        // Build subproblems
        int iPos = maxIndexUsed + 1;
        unsigned j;
        int jj;

        int ic = 0;

        Point partNadirPoint = NadirPoint;
        Point partIdealPoint = IdealPoint;

        for (jj = 0; jj < currentSettings.NumberOfObjectives; jj++) {
            j = off(offset, jj);

            if (jj > 0) {
                int j2 = off(offset, jj - 1);
                partIdealPoint.ObjectiveValues[j2] = min(IdealPoint.ObjectiveValues[j2], indexSetVec[iPivot]->ObjectiveValues[j2]);
                partNadirPoint.ObjectiveValues[j2] = NadirPoint.ObjectiveValues[j2];
            }

            int partStart = iPos;

            for (i = start; i <= end; i++) {
                if (i == iPivot)
                    continue;

                if (min(IdealPoint.ObjectiveValues[j], indexSetVec[i]->ObjectiveValues[j]) >
                    min(IdealPoint.ObjectiveValues[j], (indexSetVec)[iPivot]->ObjectiveValues[j])) {
                    indexSetVec[iPos++] = indexSetVec[i];
                }
            }

            int partEnd = iPos - 1;

            maxIndexUsed = iPos - 1;

            if (partEnd >= partStart) {
                partNadirPoint.ObjectiveValues[j] = min(IdealPoint.ObjectiveValues[j], indexSetVec[iPivot]->ObjectiveValues[j]);

                totalVolume += IQHV(partStart, partEnd, partIdealPoint, partNadirPoint, offset);
            }
        }

        maxIndexUsed = oldmaxIndexUsed;

        return (totalVolume);
    }
}
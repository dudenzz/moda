
#include "Solver.h"
#include <immintrin.h>
namespace moda {
    void Solver::EmptyCallback(int currentIteration, int totalIterations, Result *stepResult) {}
    void Solver::EmptyCallback(DataSetParameters problemSettings, Result*stepResult) {}
    void Solver::EmptyCallback(DataSetParameters problemSettings, std::string SolverMessage) {}


    void Solver::prepareData(DataSet* problem, SolverParameters settings)
    {
      
        //initialize the poblem
        currentlySolvedProblem = new DataSet(*problem);
        currentSettings = problem->getParameters();
        currentlySolvedProblem->setParameters(*currentSettings);


        //initialize solution variables

        leafs = 0;
        maxIndexUsed = 0;
        //get or calculate the better and worse point
        if (problem->typeOfOptimization == DataSet::OptimizationType::minimization)
        {
            currentlySolvedProblem->reverseObjectives();
            worsePoint = settings.GetWorseReferencePoint(currentlySolvedProblem);
            betterPoint = settings.GetBetterReferencePoint(currentlySolvedProblem);
        }
        else
        {
            worsePoint = settings.GetWorseReferencePoint(currentlySolvedProblem);
            betterPoint = settings.GetBetterReferencePoint(currentlySolvedProblem);
        }
    }


    Solver::Solver()
    {
        IterationCallback = &Solver::EmptyCallback;
        EndCallback = &Solver::EmptyCallback;
        StartCallback = &Solver::EmptyCallback;

    }
    DType Solver::Hypervolume(const Point* nadirPoint, const Point* idealPoint) {
        DType s = 1;
        short j;
        for (j = 0; j < currentSettings->NumberOfObjectives; j++) {
            
            s *= (idealPoint->ObjectiveValues[j] - nadirPoint->ObjectiveValues[j]);
        }
        return s;
    }


    DType Solver::Volume2(const Point* nadirPoint, const Point* point, const Point* idealPoint) {
        DType s = 1.0;
        const DType zero = (DType)0;
        for (short j = 0; j < currentSettings->NumberOfObjectives; j++) {
            s *= (std::min(point->ObjectiveValues[j], idealPoint->ObjectiveValues[j]) - std::max(nadirPoint->ObjectiveValues[j], zero));
        }

        return s;
    }

    DType Solver::Hypervolume(const Point *nadirPoint, const Point *p2, const Point *idealPoint)
    {

        DType s = 1;
        short j;
        for (j = 0; j < currentSettings->NumberOfObjectives; j++)
        {
            s *= (std::min(p2->ObjectiveValues[j], idealPoint->ObjectiveValues[j]) - nadirPoint->ObjectiveValues[j]);
        }
        return s;
    }




    short Solver::off(short offset, short j)
    {
        return (j + offset) % currentSettings->NumberOfObjectives;
    }

    DType Solver::IQHV(int start, int end, Point& idealPoint, Point& nadirPoint, unsigned offset) {
        if (end < start) {
            return 0;
        }

        branches++;
        offset++;
        offset = offset % currentSettings->NumberOfObjectives;

        int oldmaxIndexUsed = maxIndexUsed;

        // if there is just one point
        if (end == start) {
            return Hypervolume(&nadirPoint, currentlySolvedProblem->points[start], &idealPoint);
        }

        // if there are just two points
        if (end - start == 1) {
            DType totalVolume = Hypervolume(&nadirPoint, currentlySolvedProblem->points[start], &idealPoint);
            totalVolume += Hypervolume(&nadirPoint, currentlySolvedProblem->points[end], &idealPoint);
            tmpPoint = *(currentlySolvedProblem->points[start]);
            for (short j = 0; j < currentSettings->NumberOfObjectives; j++) {
                tmpPoint.ObjectiveValues[j] = std::min(tmpPoint.ObjectiveValues[j], currentlySolvedProblem->points[end]->ObjectiveValues[j]);
            }
            totalVolume -= Hypervolume(&nadirPoint, &tmpPoint, &idealPoint);
            return totalVolume;
        }

        unsigned iPivot = start;
        DType maxVolume = Hypervolume(&nadirPoint, (currentlySolvedProblem->points)[iPivot], &idealPoint);

        // find the pivot point
        for (unsigned i = start + 1; i <= end; i++) {
            DType volumeCurrent = Hypervolume(&nadirPoint, (currentlySolvedProblem->points)[i], &idealPoint);
            if (maxVolume < volumeCurrent) {
                maxVolume = volumeCurrent;
                iPivot = i;
            }
        }

        DType totalVolume = Hypervolume(&nadirPoint, (currentlySolvedProblem->points)[iPivot], &idealPoint);

        // build subproblems
        unsigned iPos = maxIndexUsed + 1;
        short j;
        short jj;
        int ic = 0;

        Point partNadirPoint = nadirPoint;
        Point partIdealPoint = idealPoint;

        for (jj = 0; jj < currentSettings->NumberOfObjectives; jj++) {
            j = off(offset, jj);

            if (jj > 0) {
                short j2 = off(offset, jj - 1);
                partIdealPoint.ObjectiveValues[j2] = std::min(idealPoint.ObjectiveValues[j2], currentlySolvedProblem->points[iPivot]->ObjectiveValues[j2]);
                partNadirPoint.ObjectiveValues[j2] = nadirPoint.ObjectiveValues[j2];
            }

            unsigned partStart = iPos;

            for (unsigned i = start; i <= end; i++) {
                if (i == iPivot)
                    continue;

                if (std::min(idealPoint.ObjectiveValues[j], currentlySolvedProblem->points[i]->ObjectiveValues[j]) >
                    std::min(idealPoint.ObjectiveValues[j], currentlySolvedProblem->points[iPivot]->ObjectiveValues[j])) {
                    currentlySolvedProblem->points[iPos++] = currentlySolvedProblem->points[i];
                }
            }

            unsigned partEnd = iPos - 1;
            maxIndexUsed = iPos - 1;

            if (partEnd >= partStart) {
                partNadirPoint.ObjectiveValues[j] = std::min(idealPoint.ObjectiveValues[j], currentlySolvedProblem->points[iPivot]->ObjectiveValues[j]);
                totalVolume += IQHV(partStart, partEnd, partIdealPoint, partNadirPoint, offset);
            }
        }

        maxIndexUsed = oldmaxIndexUsed;
        return totalVolume;
    }

    void Solver::RemoveDominated(int& start, int& end, Point IdealPoint)
    {
        int i;

        for (i = start; i <= end; i++)
        {
            if (currentlySolvedProblem->points[i] != NULL)
            {
                int i2;
                for (i2 = i + 1; i2 <= end; i2++)
                {
                    if (currentlySolvedProblem->points[i2] != NULL && currentlySolvedProblem->points[i] != NULL)
                    {
                        bool better = false;
                        bool worse = false;
                        short j;
                        for (j = 0; j < currentSettings->NumberOfObjectives && !(better && worse); j++)
                        {
                            DType o1 = std::min(currentlySolvedProblem->points[i]->get(j), IdealPoint[j]);

                            DType o2 = std::min(currentlySolvedProblem->points[i2]->get(j), IdealPoint[j]);

                            better = better || (o1 > o2);
                            worse = worse || (o1 < o2);
                        }
                        if (!worse)
                        {
                            // i2 covered
                            currentlySolvedProblem->points[i2] = currentlySolvedProblem->points[end--];
                        }
                        if (worse && !better)
                        {
                            // i dominated
                            currentlySolvedProblem->points[i] = currentlySolvedProblem->points[end--];
                        }
                    }
                }
            }
        }
    }
    void Solver::Normalize(std::vector <DType>& p) {
        DType nrm = Norm(p);
        DType s = 0;
        int j;
        for (j = 0; j < currentSettings->NumberOfObjectives; j++) {
            p[j] /= nrm ;
        }
    }
    DType Solver::Norm(std::vector <DType>& p) {
        DType s = 0;
        int j;
        for (j = 0; j < currentSettings->NumberOfObjectives; j++) {
            s += p[j] * p[j];
        }
        return sqrt(s);
    }

}
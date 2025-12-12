
#include <IQHVSolver.h>
#include <DataSet.h>
#include <Point.h>
#include <random>
#include <HSSSolver.h>
#include <DBHVESolver.h>
#include <MCHVESolver.h>
#include <QEHCSolver.h>
#include <HSSSolver.h>
#include <QHV_BQSolver.h>
#include <QHV_BR.h>
#include "MemoryManager.h"
#include "ostream"
#include "IQHVSolver.h"
#include "DBHVESolver.h"
#include "Result.h"
#include "../OriginalSolutions/OriginalIQHV.h"
#include "../OriginalSolutions/OriginalHVE.h"
#include "../OriginalSolutions/OriginalHSS.h"
#include "../OriginalSolutions/OriginalQEHC.h"
#include <filesystem>
#include <fstream>
#include <thread>
#define TEST_EPSILON 0.002
#define IDENTITY_EPSILON 0.000001
#define SLOWDOWN 3
using namespace moda;


void TestDatasetConversions()
{
    DataSet* ds = DataSet::LoadFromFilename("NDSets/NDRandEuclideanCheb4_100_100_0.txt");
    //NDTree<Point> ts = ds->toNDTree();
}

void TestIQHV()
{
    //In this method we create a new dataset ad hoc, using a completely randomized data. This process simulates an exterior algorithm, which generates non-dominated set of solutions for which we calculate the Hypervolume.
    //If we call the default, non-parametrized constructor, we create a container in which we can put 2-objective solutions.
    DataSet* ds = new DataSet();

    //This is how we create  dataset for n-objective problems (e.g. 5 objectives).
    int n = 5;
    ds = new DataSet(n);

    //This dataset is now empty. We need to fill it with data.
    //This data structure is capable of capturiing the data for one solution. We use a very general name for the structure - Point. The dimensionality of the point should be compatible with the dataset.
    Point* p = new Point(n);

    //At this moment the data structure is empty. This is how we can fill the data structure by hand.
    p->ObjectiveValues[0] = 0.2;
    p->ObjectiveValues[1] = 0.2;
    p->ObjectiveValues[2] = 0.2;
    p->ObjectiveValues[3] = 0.4;
    p->ObjectiveValues[4] = 0.5;

    //Now we can add this point to the dataset.
    ds->add(p);


    //Let's create a few hundred random points and add them into the dataset. This is a simulation of an algorithm, which generates solutions to be evaluated ad-hoc.
    for (int i = 0; i < 783; i++)
    {
        p = new Point(n);
        for (int j = 0; j < n; j++)
            p->ObjectiveValues[j] = (DType)rand() / RAND_MAX;
        ds->add(p);
    }

    //right now the data is not normalized, it consists of a completely random cloud of points. We can normalize the dataset (it's a min-max normalization).
    ds->normalize();

    //Let's now calculate the hypervolume. The most basic approach is to create a solver.
    IQHVSolver solver;

    //And a set of parameters. We can do it later, or while constructing the parameters object - we need to specify, how we interpret the reference points in the calculation of hypervolume. The first parameter is responsible for the 
    //definition of the "better reference point", the second one for the definition of "worse reference point". e.g. The worse reference point is the lower boundary, if we are trying to calculate hypervolume below the dataset (i.e. we are performing maximization).
    //The worse reference point is the upper boundary, if we are performing minimzation.
    IQHVParameters* params = new IQHVParameters(IQHVParameters::ReferencePointCalculationStyle::zeroone, IQHVParameters::ReferencePointCalculationStyle::exact);

    //Let's now choose the direction of calculation - are we going to calculate a hypervolume "below" or "above" the pareto front?
    ds->typeOfOptimization = DataSet::OptimizationType::maximization;

    //Finally we can calculate the hypervolume
    HypervolumeResult* result = solver.Solve(ds, *params);

    //The result contains the information on the execution time and the calculated hypervolume.
    std::cout << "Calculated Hypervolume: " << result->HyperVolume << " Elapsed time: " << result->ElapsedTime << "ms" << std::endl;
}

void TestHSS_QEHC()
{
    //Let's now load up a dataset from file and calculate the optimal subset of solutions from the set of solutions in the file.
    //We use a dedicated function to load a dataset from file.
    DataSet* ds = DataSet::LoadFromFilename("NDSets/NDRandEuclideanCheb4_100_100_0.txt");

    //we can normalize the dataset, let's make sure we are using maximization this time
    //ds->normalize();
    ds->typeOfOptimization = DataSet::OptimizationType::maximization;

    //All it takes is creating a solver and a set of parameters. 
    HSSSolver solver;
    HSSParameters* params = new HSSParameters(HSSParameters::ReferencePointCalculationStyle::exact, HSSParameters::ReferencePointCalculationStyle::exact);

    //We can specify wether we want to use the incremental or the decremental solution and the number of points we are looking for.
    params->StoppingSubsetSize = 20;
    params->Strategy = HSSParameters::SubsetSelectionStrategy::Incremental;

    //let's find the optimal subset and print out it's hypervolume.
    HSSResult* result = solver.Solve(ds, *params);
    std::cout << "Calculated hypervolume for 20 optimal solutions (incremental): " << result->HyperVolume << std::endl;

    //let's increase the number of points and calculate it again
    params->StoppingSubsetSize = 50;
    result = solver.Solve(ds, *params);
    std::cout << "Calculated hypervolume for 50 optimal solutions (incremental): " << result->HyperVolume << std::endl;

    //let's now compare it to the total hypervolume of this exact dataset
    IQHVSolver hv_solver;
    IQHVParameters* hv_params = new IQHVParameters(IQHVParameters::ReferencePointCalculationStyle::exact, IQHVParameters::ReferencePointCalculationStyle::exact);
    HypervolumeResult* hv_result = hv_solver.Solve(ds, *hv_params);
    std::cout << "Calculated hypervolume for all solutions: " << hv_result->HyperVolume << std::endl;

    //let's now change the method to decremental; since there are a couple of thousands points in the dataset, we are expecting higher execution time for those exact parameters
    params->StoppingSubsetSize = 50;
    params->Strategy = HSSParameters::SubsetSelectionStrategy::Incremental;

    //let's find the optimal subset and print out it's hypervolume.
    result = solver.Solve(ds, *params);
    std::cout << "Calculated hypervolume for 50 optimal solutions (decremental): " << result->HyperVolume << std::endl;

    //let's increase the number of points and calculate it again
    params->StoppingSubsetSize = 20;
    result = solver.Solve(ds, *params);
    std::cout << "Calculated hypervolume for 20 optimal solutions (decremental): " << result->HyperVolume << std::endl;


    //This is a subject to work in progress

    //In this test we are also going to test the QEHCSolver solver, which calculates the points with the highest and lowest contribution in the entire dataset. Keep in mind, that the current implementation of the particular method uses a very large
    //default size of vectors (it allocates A LOT of memory) - for small problems, this could prove to be an issue. We are actively working on the solution.
    QEHCSolver qehc_solver;
    QEHCParameters* qehc_params = new QEHCParameters(QEHCParameters::ReferencePointCalculationStyle::exact, QEHCParameters::ReferencePointCalculationStyle::exact);

    //we are looking for a point with maximum contribution
    qehc_params->SearchSubject = QEHCParameters::SearchSubjectOption::MaximumContribution;
    params->Strategy = HSSParameters::SubsetSelectionStrategy::Incremental;
    result = solver.Solve(ds, *params);
    QEHCResult* qehc_result = qehc_solver.Solve(ds, *qehc_params);

    //let's now check wether the selected point is within the 20 selected points by HSSSolver
    std::stringstream subset_as_string;
    std::copy(result->selectedPoints.begin(), result->selectedPoints.end(), std::ostream_iterator<int>(subset_as_string, " "));
    std::cout << "{" << qehc_result->MaximumContributionIndex << "} in {" << subset_as_string.str() << "}" << std::endl;
}

void TestHVE()
{
    //Let's move on to hypervolume estimation. Here we are going to use another type of loading the dataset - we are going to load all datasets in a specific directory.
    auto all_datasets = DataSet::LoadBulk("NDSets");

    //we are going to iterate over all of the datasets until we approach one, which has 6 objectives; once we are done with it, we are going to break the iteration
    for (DataSet* ds : all_datasets)
    {
        if (ds->getParameters()->NumberOfObjectives == 4)
        {
            //Let's calculate the total hypervolume for this dataset.
            ds->normalize();
            ds->typeOfOptimization = DataSet::OptimizationType::minimization;
            IQHVSolver hv_solver;
            IQHVParameters* hv_params = new IQHVParameters(IQHVParameters::ReferencePointCalculationStyle::zeroone, IQHVParameters::ReferencePointCalculationStyle::exact);
            DType hypervolume = hv_solver.Solve(ds, *hv_params)->HyperVolume;

            //We are going to use a couple of different estimation methods: HVE, QHV_BQSolver and MCHV. They all share the same interface.
            MCHVESolver mc_solver;
            MCHVParameters* mc_params = new MCHVParameters(MCHVParameters::ReferencePointCalculationStyle::zeroone, MCHVParameters::ReferencePointCalculationStyle::exact);
            mc_params->MaxEstimationTime = 300;
            MCHVResult* mc_result = mc_solver.Solve(ds, *mc_params);
            DBHVESolver hve_solver;
            DBHVEParameters* hve_params = new DBHVEParameters(DBHVEParameters::ReferencePointCalculationStyle::zeroone, DBHVEParameters::ReferencePointCalculationStyle::exact);
            hve_params->MaxEstimationTime = 300;
            DBHVEResult* hve_result = hve_solver.Solve(ds, *hve_params);
            QHV_BQSolver bq_solver;
            QHV_BQParameters* bq_params = new QHV_BQParameters(QHV_BQParameters::ReferencePointCalculationStyle::zeroone, QHV_BQParameters::ReferencePointCalculationStyle::exact);
            bq_params->MonteCarlo = false;
            bq_params->MaxEstimationTime = 300;
            QHV_BQResult* bq_result = bq_solver.Solve(ds, *bq_params);

            //each algorithm had 300 milliseconds for estimation - still, keep in mind the memory allocation is not considered estimation time, it is quite long - a problem we are actively working on.
            //HVE estimates the hypervolume value; while MCHV and QHV_BQSolver provide upper and lower boundaries
            //let's analyze the results
            std::cout << "Exact hypervolume: " << hypervolume << std::endl;
            std::cout << "HVE Estimation (300ms): " << hve_result->HyperVolumeEstimation << std::endl;
            std::cout << "MCHV Estimation (300ms): " << bq_result->LowerBound << "<" << hypervolume << "<" << bq_result->UpperBound << std::endl;
            std::cout << "QHV_BQ Estimation (300ms): " << mc_result->LowerBound << "<" << hypervolume << "<" << mc_result->UpperBound << std::endl;
            break;
        }
    }
}

void IterationCallback(int currentIteration, int totalIterations, Result* stepResult)
{
    //so, this is the default iteration callback, it provides us with current iteration, total number of iterations and current, contemporary result
    //as the callback is uniform for all solver types, we should cast the result - the types are compatibile with each other.
    HSSResult* hss_result = (HSSResult*)stepResult;

    //Lets create a string for the selected subset
    std::stringstream subset_as_string;
    std::copy(hss_result->selectedPoints.begin(), hss_result->selectedPoints.end(), std::ostream_iterator<int>(subset_as_string, " "));

    //it is very often convenient to use carriage return to end the callback feedback to the standard output, but it's not very good for the presentation sake
    std::cout << currentIteration << "/" << totalIterations << " {" << subset_as_string.str() << "}" << std::endl;

}
void TestCallbacks()
{
    //let's now imagine we want to follow the process of solving a problem. This solution is uniform for all algorithm, but we will try it out on the HSSSolver example.
    //we start as usual, but this time, on purpose I am choosing a harder dataset (so we can observe the callbacks)
    DataSet* ds = DataSet::LoadFromFilename("NDSets/NDRandEuclideanCheb4_100_100_0.txt");
    ds->typeOfOptimization = DataSet::OptimizationType::maximization;
    HSSSolver solver;
    solver.IterationCallback = IterationCallback;

    //once the solver is created, we can appoint a callback to the solver, let's create one
    HSSParameters* params = new HSSParameters(HSSParameters::ReferencePointCalculationStyle::exact, HSSParameters::ReferencePointCalculationStyle::exact);
    params->StoppingSubsetSize = 10;
    params->Strategy = HSSParameters::SubsetSelectionStrategy::Incremental;

    //additionally we should specify that we want to launch callbacks
    params->callbacks = true;

    //that's it, now I can follow the process of adding points 
    HSSResult* result = solver.Solve(ds, *params);
    std::cout << "Calculated hypervolume for 10 optimal solutions (incremental): " << result->HyperVolume << std::endl;
}

void TestNDSets()
{
    auto all_problems = DataSet::LoadBulk("NDSets/");

    for (auto problem : all_problems) {
        //        if (problem->getParameters()->NumberOfObjectives > 6 && problem->points.size() > 1) {

        problem->normalize();

        problem->reverseObjectives();

        NDTree<Point> NDTree;

        int ii;
        for (ii = 0; ii < problem->points.size(); ii++) {
            NDTree.update(*problem->points[ii], false);
        }
        std::cout << problem->filename << ' ' << NDTree.numberOfSolutions() << ' ';

        DType currentMin = 1e30;
        Point ReferencePoint;
        for (int j = 0; j < problem->getParameters()->NumberOfObjectives; j++) {
            ReferencePoint.ObjectiveValues[j] = 1.0;
        }
        std::vector <DType> WeightVector;
        WeightVector.resize(problem->getParameters()->NumberOfObjectives, 1.0);

        BestSolution bestSolution;

        if (problem->points.size() == 1) {
            std::cout << 'x';
        }

        NDTree.root->minScalarizingFunction(currentMin, ReferencePoint, WeightVector, bestSolution);

        std::cout << currentMin << ' ';
        for (int j = 0; j < problem->getParameters()->NumberOfObjectives; j++) {
            std::cout << bestSolution.solution->ObjectiveValues[j] << ' ';
        }

        DBHVESolver hve;
        DBHVEParameters settings(SolverParameters::ReferencePointCalculationStyle::exact, SolverParameters::ReferencePointCalculationStyle::exact);
        settings.MCiterations = 10000;

        //            problem->setNormalize(true);
        problem->typeOfOptimization = DataSet::OptimizationType::minimization;

        BoundedResult* hveRes = hve.Solve(problem, settings);

        std::cout << problem->filename << ' ' << hveRes->HyperVolumeEstimation << ' ' << problem->points.size() << ' ';

        IQHVSolver iqhv;

        IQHVParameters* iqhv_params = new IQHVParameters(SolverParameters::ReferencePointCalculationStyle::exact, SolverParameters::ReferencePointCalculationStyle::exact);
        iqhv_params->callbacks = false;

        HypervolumeResult* iqhv_res = iqhv.Solve(problem, *iqhv_params);
        std::cout << ' ' << iqhv_res->HyperVolume << ' ';

        MCHVESolver mchv;

        MCHVParameters* mchv_params = new MCHVParameters(SolverParameters::ReferencePointCalculationStyle::exact, SolverParameters::ReferencePointCalculationStyle::exact);

        mchv_params->callbacks = false;
        mchv_params->MaxEstimationTime = 1000000;

        auto mchv_res = mchv.Solve(problem, *mchv_params);

        std::cout << ' ' << mchv_res->HyperVolumeEstimation << ' ';

        QHV_BQSolver qhvbq;

        QHV_BQParameters* qhvbq_params = new QHV_BQParameters(SolverParameters::ReferencePointCalculationStyle::exact, SolverParameters::ReferencePointCalculationStyle::exact);

        qhvbq_params->callbacks = false;
        qhvbq_params->MaxEstimationTime = 10;

        auto qhvbq_res = qhvbq.Solve(problem, *qhvbq_params);

        std::cout << ' ' << qhvbq_res->LowerBound << ' ' << qhvbq_res->UpperBound << ' ';

        std::cout << '\n';
    }

}


int testReverseNDSets()
{
    auto all_problems = DataSet::LoadBulk("NdSets/");
    //auto all_problems = DataSet::LoadBulk("C:/hypervolume - original/QHV-BQ/QHV-BQ/source-code/linear/max", false);

    for (auto problem : all_problems) {
        if (problem->getParameters()->NumberOfObjectives == 4 /* && problem->points.size() > 1*/) {
            problem->typeOfOptimization = DataSet::OptimizationType::maximization;
            //problem->reverseObjectives();
            problem->normalize();

            NDTree<Point> NDTree;
            Point* worsePoint = new Point();

            for (int j = 0; j < problem->getParameters()->NumberOfObjectives; j++) {
                worsePoint->ObjectiveValues[j] = 0;
            }
            std::cout << problem->getParameters()->nPoints << " ";
            int ii;
            for (ii = 0; ii < problem->points.size(); ii++) {
                NDTree.update(*problem->points[ii], true);
            }
            std::cout << problem->getParameters()->NumberOfObjectives << ' ' << problem->filename << ' ' << NDTree.numberOfSolutions() << ' ';

            if (problem->points.size() == 1) {
                std::cout << 'x';
            }

            DBHVESolver hve;
            DBHVEParameters settings(SolverParameters::ReferencePointCalculationStyle::zeroone, SolverParameters::ReferencePointCalculationStyle::zeroone);

            settings.worseReferencePoint = worsePoint;
            settings.MCiterations = 100000;
            settings.seed = 4241;
            BoundedResult* hveRes = hve.Solve(problem, settings);

            std::cout << problem->filename << ' ' << hveRes->HyperVolumeEstimation << ' ' << problem->points.size() << ' ';


            if (problem->getParameters()->NumberOfObjectives < 6) {
                IQHVSolver iqhv;
                IQHVParameters* iqhv_params = new IQHVParameters(SolverParameters::ReferencePointCalculationStyle::zeroone, SolverParameters::ReferencePointCalculationStyle::zeroone);
                iqhv_params->callbacks = false;
                iqhv_params->worseReferencePoint = worsePoint;

                HypervolumeResult* iqhv_res = iqhv.Solve(problem, *iqhv_params);
                std::cout << ' ' << iqhv_res->HyperVolume << ' ' << iqhv_res->ElapsedTime;
            }

            std::cout << '\n';
        }
    }

    return 1;

}

void DBHVETimeTestsCompiled(std::string filepath)
{
    auto ds = DataSet::LoadFromFilename(filepath);
    ds->typeOfOptimization = DataSet::OptimizationType::maximization;
    ds->setName(ds->filename);
    ds->normalize();
    moda::scalarizingCalls = 0;
    OriginalHVE::scalarizingCalls = 0;
    OriginalHVE::NumberOfObjectives = ds->points[0]->NumberOfObjectives;
    OriginalHVE::sumDominatingFactor = pow(PI, OriginalHVE::NumberOfObjectives / 2.0) / tgamma(OriginalHVE::NumberOfObjectives / 2.0 + 1) * pow(0.5, OriginalHVE::NumberOfObjectives) /
        MONTE_CARLO_ITERATIONS;
    //if (OriginalIQHV::NumberOfObjectives != 4) return;
    int nSol = ds->points.size();
    DBHVESolver dbSolver;
    DBHVEParameters* dbParams = new DBHVEParameters(SolverParameters::exact, SolverParameters::exact);
    dbParams->MaxEstimationTime = 10000;
    dbParams->MCiterations = 100000;
    dbParams->seed = 0;

    IQHVParameters* params = new IQHVParameters(SolverParameters::exact, SolverParameters::exact);
    DBHVEResult* result2 = dbSolver.Solve(ds, *dbParams);
    //return;
    std::vector <OriginalHVE::TPoint*> allSolutions;
    std::ostringstream fileName;
    fileName << filepath;
    std::fstream Stream(fileName.str(), std::ios::in);
    allSolutions.clear();
    OriginalHVE::Load(allSolutions, Stream);
    Stream.close();
    ds->points.clear();
    delete ds;
    delete params;

    OriginalHVE::TPoint idealPoint, nadirPoint;
    int ii, jj;
    for (jj = 0; jj < OriginalHVE::NumberOfObjectives; jj++) {
        idealPoint.ObjectiveValues[jj] = allSolutions[0]->ObjectiveValues[jj];
        nadirPoint.ObjectiveValues[jj] = allSolutions[0]->ObjectiveValues[jj];
    }
    for (ii = 1; ii < allSolutions.size(); ii++) {
        for (jj = 0; jj < OriginalHVE::NumberOfObjectives; jj++) {
            if (idealPoint.ObjectiveValues[jj] < allSolutions[ii]->ObjectiveValues[jj]) {
                idealPoint.ObjectiveValues[jj] = allSolutions[ii]->ObjectiveValues[jj];
            }
            if (nadirPoint.ObjectiveValues[jj] > allSolutions[ii]->ObjectiveValues[jj]) {
                nadirPoint.ObjectiveValues[jj] = allSolutions[ii]->ObjectiveValues[jj];
            }
        }
    }
    OriginalHVE::normalize(allSolutions, idealPoint, nadirPoint);

    for (jj = 0; jj < OriginalIQHV::NumberOfObjectives; jj++) {
        idealPoint.ObjectiveValues[jj] = 1;
        nadirPoint.ObjectiveValues[jj] = 0;
    }
    clock_t start = clock();
    double result = OriginalHVE::approximateHVMax(allSolutions, idealPoint, nadirPoint, nSol, false, 0, false);
    clock_t elapsed = clock() - start;

    double current_slowdown = result2->ElapsedTime / (double)elapsed;
    OriginalIQHV::indexSetVec.clear();
    OriginalIQHV::indexSetVec.resize(0);
    std::cout << "--- DBHVE evaluation ---\n";
    std::cout << "File: " << filepath << std::endl;
    std::cout << "original solution: " << result << std::endl;
    std::cout << "new solution: " << result2->HyperVolumeEstimation << std::endl;
    std::cout << "Time ratio(slowdown): " << std::to_string(current_slowdown).c_str();
    std::cout << " \noriginal time: ";
    std::cout << std::to_string(elapsed).c_str();
    std::cout << " \nnew time: ";
    std::cout << std::to_string(result2->ElapsedTime).c_str();
    std::cout << "\n";
    std::cout << "Scalarizing calls in original: " << OriginalHVE::scalarizingCalls;
    std::cout << "\n";
    std::cout << "Scalarizing calls in moda: " << moda::scalarizingCalls;
    std::cout << "\n";
    std::cout << "----------------------------------------------------\n\n";
}

void HSSTimeTestsCompiled(std::string filepath)
{
    auto ds = DataSet::LoadFromFilename(filepath);
    ds->typeOfOptimization = DataSet::OptimizationType::maximization;
    ds->setName(ds->filename);
    ds->normalize();
    OriginalHSS::NumberOfObjectives = ds->points[0]->NumberOfObjectives;

    //if (OriginalHSS::NumberOfObjectives != 4) return;
    int nSol = ds->points.size();
    HSSSolver hssSolver;
    HSSParameters* hssParams = new HSSParameters(SolverParameters::exact, SolverParameters::exact);
    hssParams->StoppingSubsetSize = 10;
    hssParams->StoppingCriteria = HSSParameters::StoppingCriteriaType::SubsetSize;
    hssParams->Strategy = HSSParameters::SubsetSelectionStrategy::Decremental;


    IQHVParameters* params = new IQHVParameters(SolverParameters::exact, SolverParameters::exact);

    HSSResult* hssResult = hssSolver.Solve(ds, *hssParams);
    //return;
    std::vector <OriginalHSS::TPoint*> allSolutions;
    std::ostringstream fileName;
    fileName << filepath;
    std::fstream Stream(fileName.str(), std::ios::in);
    allSolutions.clear();
    OriginalHSS::Load(allSolutions, Stream);
    Stream.close();
    ds->points.clear();
    delete ds;
    delete params;

    OriginalHSS::TPoint idealPoint, nadirPoint;
    int ii, jj;
    for (jj = 0; jj < OriginalHSS::NumberOfObjectives; jj++) {
        idealPoint.ObjectiveValues[jj] = allSolutions[0]->ObjectiveValues[jj];
        nadirPoint.ObjectiveValues[jj] = allSolutions[0]->ObjectiveValues[jj];
    }
    for (ii = 1; ii < allSolutions.size(); ii++) {
        for (jj = 0; jj < OriginalHSS::NumberOfObjectives; jj++) {
            if (idealPoint.ObjectiveValues[jj] < allSolutions[ii]->ObjectiveValues[jj]) {
                idealPoint.ObjectiveValues[jj] = allSolutions[ii]->ObjectiveValues[jj];
            }
            if (nadirPoint.ObjectiveValues[jj] > allSolutions[ii]->ObjectiveValues[jj]) {
                nadirPoint.ObjectiveValues[jj] = allSolutions[ii]->ObjectiveValues[jj];
            }
        }
    }
    OriginalHSS::normalize(allSolutions, idealPoint, nadirPoint);

    for (jj = 0; jj < OriginalHSS::NumberOfObjectives; jj++) {
        idealPoint.ObjectiveValues[jj] = 1;
        nadirPoint.ObjectiveValues[jj] = 0;
    }
    clock_t start = clock();
    vector <int> selectedPointsIds;
    OriginalHSS::greedyHSSDecLazyQHV_II(allSolutions, selectedPointsIds, idealPoint, nadirPoint, 10);
    vector <OriginalHSS::TPoint*> selectedPoints;
    for (int i = 0; i < 10; i++)
    {
        selectedPoints.push_back(allSolutions[selectedPointsIds[i]]);
    }
    double result = OriginalHSS::solveQHV_II(selectedPoints, idealPoint, nadirPoint);
    clock_t elapsed = clock() - start;

    double current_slowdown = hssResult->ElapsedTime / (double)elapsed;
    OriginalHSS::indexSetVec.clear();
    OriginalHSS::indexSetVec.resize(0);
    std::cout << "--- HSS evaluation ---\n";
    std::cout << "File: " << filepath << std::endl;
    std::cout << "original solution: " << result << std::endl;
    std::cout << "new solution: " << hssResult->HyperVolume << std::endl;
    std::cout << "Time ratio(slowdown): " << std::to_string(current_slowdown).c_str();
    std::cout << " \noriginal time: ";
    std::cout << std::to_string(elapsed).c_str();
    std::cout << " \nnew time: ";
    std::cout << std::to_string(hssResult->ElapsedTime).c_str();
    std::cout << "\n";

    std::cout << "----------------------------------------------------\n\n";
}

void IQHVTimeTestsCompiled(std::string filepath)
{
    auto ds = DataSet::LoadFromFilename(filepath);
    if (ds->getParameters()->NumberOfObjectives == 10) return;
    std::cout << "--- IQHV evaluation ---\n";
    std::cout << "File: " << filepath << std::endl;
    ds->typeOfOptimization = DataSet::OptimizationType::maximization;
    ds->setName(ds->filename);
    ds->normalize();
    OriginalIQHV::NumberOfObjectives = ds->points[0]->NumberOfObjectives;

    int nSol = ds->points.size();
    IQHVSolver solver;
    DBHVESolver dbSolver;
    DBHVEParameters* dbParams = new DBHVEParameters(SolverParameters::exact, SolverParameters::exact);
    dbParams->MaxEstimationTime = 100000;
    dbParams->MCiterations = 5000;
    dbParams->seed = 0;

    IQHVParameters* params = new IQHVParameters(SolverParameters::exact, SolverParameters::exact);

    HypervolumeResult* result_lib = solver.Solve(ds, *params);

    //return;
    DBHVEResult* estimation = dbSolver.Solve(ds, *dbParams);
    std::vector <OriginalIQHV::TPoint*> allSolutions;
    allSolutions.resize(1000);
    std::ostringstream fileName;
    fileName << filepath;
    std::fstream Stream(fileName.str(), std::ios::in);
    allSolutions.clear();
    OriginalIQHV::Load(allSolutions, Stream);
    int iter = 0;
    Stream.close();
    ds->points.clear();
    delete ds;
    delete params;

    OriginalIQHV::TPoint idealPoint, nadirPoint;
    int ii, jj;
    for (jj = 0; jj < OriginalIQHV::NumberOfObjectives; jj++) {
        idealPoint.ObjectiveValues[jj] = allSolutions[0]->ObjectiveValues[jj];
        nadirPoint.ObjectiveValues[jj] = allSolutions[0]->ObjectiveValues[jj];
    }
    for (ii = 1; ii < allSolutions.size(); ii++) {
        for (jj = 0; jj < OriginalIQHV::NumberOfObjectives; jj++) {
            if (idealPoint.ObjectiveValues[jj] < allSolutions[ii]->ObjectiveValues[jj]) {
                idealPoint.ObjectiveValues[jj] = allSolutions[ii]->ObjectiveValues[jj];
            }
            if (nadirPoint.ObjectiveValues[jj] > allSolutions[ii]->ObjectiveValues[jj]) {
                nadirPoint.ObjectiveValues[jj] = allSolutions[ii]->ObjectiveValues[jj];
            }
        }
    }
    OriginalIQHV::normalize(allSolutions, idealPoint, nadirPoint);

    for (jj = 0; jj < OriginalIQHV::NumberOfObjectives; jj++) {
        idealPoint.ObjectiveValues[jj] = 1;
        nadirPoint.ObjectiveValues[jj] = 0;
    }
    clock_t start = clock();

    double result = OriginalIQHV::solveQHV_II(allSolutions, idealPoint, nadirPoint, nSol);
    clock_t elapsed = clock() - start;


    OriginalIQHV::indexSetVec.clear();
    OriginalIQHV::indexSetVec.resize(0);

    std::cout << "original solution: " << result << std::endl;

    std::cout << "new solution: " << result_lib->HyperVolume << std::endl;

    std::cout << "estimation: " << estimation->HyperVolumeEstimation << std::endl;
    double current_slowdown = result_lib->ElapsedTime / (double)elapsed;
    std::cout << "\nTime ratio: " << std::to_string(current_slowdown).c_str();
    std::cout << " \noriginal time: ";
    std::cout << std::to_string(elapsed).c_str();
    std::cout << " \nnew time: ";
    std::cout << std::to_string(result_lib->ElapsedTime).c_str();

    std::cout << " \nestimation time: ";
    std::cout << std::to_string(estimation->ElapsedTime).c_str();
    std::cout << "\n";

    std::cout << "----------------------------------------------------\n\n";
}


void QEHCTimeTestsCompiled(std::string filepath)
{
    auto ds = DataSet::LoadFromFilename(filepath);
    ds->typeOfOptimization = DataSet::OptimizationType::maximization;
    ds->setName(ds->filename);
    OriginalQEHC::NumberOfObjectives = ds->points[0]->NumberOfObjectives;

    if (OriginalQEHC::NumberOfObjectives > 7) return;// || ds->points.size() != 100) return;
    int nSol = ds->points.size();
    QEHCSolver qehcSolver;
    QEHCParameters* qehcParams = new QEHCParameters(SolverParameters::exact, SolverParameters::exact);
    qehcParams->sort = false;
    qehcParams->sort = false;
    qehcParams->maxlevel = 10;
    qehcParams->SearchSubject = QEHCParameters::SearchSubjectOption::MinimumContribution;
    QEHCResult* qehcResult = qehcSolver.Solve(ds, *qehcParams);
    //IQHVSolver iqSolver;
    //IQHVParameters* iqhvParams = new IQHVParameters(SolverParameters::exact, SolverParameters::exact);
    //double totalVol = iqSolver.Solve(ds, *iqhvParams)->HyperVolume;
    double minContr = 100;
    int minId = 0;
    clock_t raw_start = clock();
    for (int i = 0; i < nSol; i++)
    {
        //std::cout << "\r" << i << "/" << 99;
        //std::vector<int> vec2(ds->points.size() - 1);
        //DataSet* ids = new DataSet(ds->getParameters()->NumberOfObjectives);
        //ids->points.resize(99);
        //ids->setNumberOfPoints(99);
        //// Copy segment 1
        //for (size_t k = 0; k < i; ++k) {
        //    ids->points[k] = new Point(*ds->points[k]); // DEEP COPY
        //}

        //// Copy segment 2
        //for (size_t k = i + 1; k < ds->points.size(); ++k) {
        //    // k - 1 accounts for the skipped index i
        //    ids->points[k - 1] = new Point(*ds->points[k]); // DEEP COPY
        //}
        //double newSol = iqSolver.Solve(ids, *iqhvParams)->HyperVolume;
        //double contribution = totalVol - newSol;
        //if (contribution < minContr)
        //{
        //    minContr = contribution;
        //    minId = i;
        //}
        //delete ids;
    }
    std::cout << "\r";
    clock_t raw_end = clock();
    //return;
    std::vector <OriginalQEHC::TPoint*> allSolutions;
    std::ostringstream fileName;
    fileName << filepath;
    std::fstream Stream(fileName.str(), std::ios::in);
    allSolutions.clear();
    OriginalQEHC::Load(allSolutions, Stream);
    Stream.close();
    ds->points.clear();
    delete ds;

    OriginalQEHC::TPoint idealPoint, nadirPoint;
    int ii, jj;
    for (jj = 0; jj < OriginalQEHC::NumberOfObjectives; jj++) {
        idealPoint.ObjectiveValues[jj] = allSolutions[0]->ObjectiveValues[jj];
        nadirPoint.ObjectiveValues[jj] = allSolutions[0]->ObjectiveValues[jj];
    }
    for (ii = 1; ii < allSolutions.size(); ii++) {
        for (jj = 0; jj < OriginalQEHC::NumberOfObjectives; jj++) {
            if (idealPoint.ObjectiveValues[jj] < allSolutions[ii]->ObjectiveValues[jj]) {
                idealPoint.ObjectiveValues[jj] = allSolutions[ii]->ObjectiveValues[jj];
            }
            if (nadirPoint.ObjectiveValues[jj] > allSolutions[ii]->ObjectiveValues[jj]) {
                nadirPoint.ObjectiveValues[jj] = allSolutions[ii]->ObjectiveValues[jj];
            }
        }
    }


    for (jj = 0; jj < OriginalQEHC::NumberOfObjectives; jj++) {
        idealPoint.ObjectiveValues[jj] = 1;
        nadirPoint.ObjectiveValues[jj] = 0;
    }
    clock_t start = clock();
    clock_t tempClock;
    int processId = -1;
    double result = OriginalQEHC::solveQEHC(allSolutions, nSol, true, tempClock, tempClock, processId, true, true);
    clock_t elapsed = clock() - start;

    double current_slowdown = qehcResult->ElapsedTime / (double)elapsed;
    OriginalQEHC::indexSetVec.clear();
    OriginalQEHC::indexSetVec.resize(0);
    std::cout << "--- QEHC evaluation ---\n";
    std::cout << "File: " << filepath << std::endl;
    //std::cout << "raw result: " << minContr << ":" << minId << std::endl;
    //std::cout << "raw time: " << raw_end - raw_start << std::endl;
    std::cout << "original solution: " << result << ":" << processId << std::endl;
    std::cout << "new solution: " << qehcResult->MinimumContribution << ":" << qehcResult->MinimumContributionIndex << std::endl;
    std::cout << "Time ratio(slowdown): " << std::to_string(current_slowdown).c_str();
    std::cout << " \noriginal time: ";
    std::cout << std::to_string(elapsed).c_str();
    std::cout << " \nnew time: ";
    std::cout << std::to_string(qehcResult->ElapsedTime).c_str();
    std::cout << "\n";

    std::cout << "----------------------------------------------------\n\n";
}

void QEHCMaxlevelTimeTestsCompiled(std::string filepath)
{
    auto ds = DataSet::LoadFromFilename(filepath);
    ds->typeOfOptimization = DataSet::OptimizationType::maximization;
    ds->setName(ds->filename);
    if (ds->getParameters()->NumberOfObjectives != 8) return;
    int nSol = ds->points.size();
    vector<int> levels = { 1,3,5,7,10,15,20,25,30,50,100,200,500,1000,5000,10000 };
    OriginalQEHC::NumberOfObjectives = ds->points[0]->NumberOfObjectives;


    std::vector <OriginalQEHC::TPoint*> allSolutions;
    std::ostringstream fileName;
    fileName << filepath;
    std::fstream Stream(fileName.str(), std::ios::in);
    allSolutions.clear();
    OriginalQEHC::Load(allSolutions, Stream);
    Stream.close();

    for (auto mlvl : levels)
    {
        QEHCSolver qehcSolver;
        QEHCParameters* qehcParams = new QEHCParameters(SolverParameters::exact, SolverParameters::exact);
        qehcParams->sort = false;
        qehcParams->maxlevel = mlvl;
        qehcParams->SearchSubject = QEHCParameters::SearchSubjectOption::MinimumContribution;
        QEHCResult* qehcResult = qehcSolver.Solve(ds, *qehcParams);


        clock_t start = clock();
        clock_t tempClock;
        int processId = -1;
        double result = OriginalQEHC::solveQEHC(allSolutions, nSol, true, tempClock, tempClock, processId, true, true);
        clock_t elapsed = clock() - start;

        std::cout << filepath.substr(filepath.rfind("\\") + 1) << "\t" << ds->getParameters()->nPoints << "\t" << ds->getParameters()->NumberOfObjectives << "\t" << mlvl << "\t" << std::to_string(qehcResult->ElapsedTime).c_str() << "\t" << elapsed << "\n";
        //return;
        //std::cout << "--- QEHC evaluation ---\n";
        //std::cout << "File: " << filepath << std::endl;
        //std::cout << " \ttime: ";
        //std::cout << std::to_string(qehcResult->ElapsedTime).c_str();
        //std::cout << "\n";
        //std::cout << "----------------------------------------------------\n\n";
    }
    ds->points.clear();
    delete ds;
}
int main(int argc, char* argv[])
{

    //TestIQHV();
    //TestHSS_QEHC();
    //TestHVE();
    //TestCallbacks();
    //TestDatasetConversions();
    //TestNDSets();
    //testReverseNDSets();
    auto p = std::filesystem::path(argv[1]);
    bool ex = std::filesystem::exists(p);
    for (const auto& entry : std::filesystem::directory_iterator(argv[1]))
    {
        if (entry.is_directory()) continue;

        auto file = entry.path().string();
        //try {
            //IQHVTimeTestsCompiled(file);
            //DBHVETimeTestsCompiled(file);
            //HSSTimeTestsCompiled(file);
        QEHCTimeTestsCompiled(file);
        //QEHCMaxlevelTimeTestsCompiled(file);
    //}
    //catch (const std::exception& exception)
    //{
    //    std::cout << "File: " << file << std::endl;
    //    std::cout << "error, file skipped. Exception message: " << std::endl << exception.what() << std::endl;
    //    std::cout << "----------------------------------------------------\n\n";
    //}
    }
    return 1;


}

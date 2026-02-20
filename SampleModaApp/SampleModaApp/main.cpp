// SampleModaProgram.cpp
//

#include <iostream>
#include <QEHCSolver.h>
#include <IQHVSolver.h>
#include <DataSet.h>
#include <HSSSolver.h>

using namespace moda;
int main()
{
 /*
    // 1. Load Data
    // Loads sample data from file (file must be accessible in execution directory)
    DataSet* ds = moda::DataSet::LoadFromFilename("../../sample-file/linear_d4n100_1");
    // Display data points one by one
    std::cout << "Loaded Data Points:\n";
    for (auto p : ds->points)
    {
        for (int i = 0; i < ds->getParameters()->NumberOfObjectives; i++)
            std::cout << p->ObjectiveValues[i] << " ";
        std::cout << "\n";
    }
    std::cout << "---------------------------------\n";
    // 2. Prepare Data and Solver
    ds->typeOfOptimization = moda::DataSet::minimization; // Set optimization type
    ds->normalize();                                      // Apply normalization
    moda::HSSSolver solver;
    moda::HSSParameters params;
    // 3. Set Solver Parameters
    // Sets calculation formulas for the reference points (zeroone substitues zeroes and ones vectors for reference points)
    using RefStyle = moda::HSSParameters::ReferencePointCalculationStyle;
    params.BetterReferencePointCalculationStyle = RefStyle::zeroone;
    params.WorseReferencePointCalculationStyle = RefStyle::zeroone;
    params.StoppingSubsetSize = 10;
    // 4. Run Solve and Display Result
    // Solve returns a pointer to a HypervolumeResult
    moda::HSSResult* result = solver.Solve(ds, params);
    std::cout << "Calculated Hypervolume: " << result->HyperVolume << std::endl;
    std::cout << "Elapsed Time (ms): " << result->ElapsedTime << std::endl;
    // Cleanup (optional but recommended)
    delete ds;
    delete result;
    */
    double** vals = new double* [16] {
        new double[2] { 0.23284221, 0.32681638 },
            new double[2] { 0.22963942, 0.32911968 },
            new double[2] { 0.05097348, 0.71530732 },
            new double[2] { 0.10222314, 0.38812172 },
            new double[2] { 0.05777225, 0.49567605 },
            new double[2] { 0.09436084, 0.47497879 },
            new double[2] { 0.08823659, 0.48089614 },
            new double[2] { 0.02708682, 1.00000000 },
            new double[2] { 0.26223131, 0.26335746 },
            new double[2] { 0.31101276, 0.15427988 },
            new double[2] { 0.30882459, 0.17954305 },
            new double[2] { 0.33337283, 0.13855275 },
            new double[2] { 0.43073651, 0.07245694 },
            new double[2] { 0.45980630, 0.06050883 },
            new double[2] { 0.90403955, 0.03170205 },
            new double[2] { 0.69166411, 0.05259329 },
        };
    DataSet* ds = new DataSet(2);
    
    for (int i = 0; i < 16; i++)
    {
        Point* p = new Point(2);
        (*p)[0] = vals[i][0];
        (*p)[1] = vals[i][1];
        ds->add(p);
    }
    ds->typeOfOptimization = DataSet::minimization;
    //ds->normalize();
    for (int i = 0; i < 16; i++)
    {
        Point p = (*ds)[i];
        (*ds)[i][0] = p[0];
        (*ds)[i][1] = p[1];
    }
    Point* nadir = new Point(2);
    (*nadir)[0] = -2;
    (*nadir)[1] = -2;
    Point* ideal = new Point(2);
    (*ideal)[0] = 1000;
    (*ideal)[1] = 1000;
    QEHCParameters* params = new QEHCParameters(SolverParameters::ReferencePointCalculationStyle::userdefined, SolverParameters::ReferencePointCalculationStyle::userdefined);
    params->worseReferencePoint = nadir;
    params->betterReferencePoint = ideal;
    QEHCSolver solver;
    //ds->typeOfOptimization = ds->minimization;
    IQHVParameters* hvparams = new IQHVParameters(SolverParameters::ReferencePointCalculationStyle::userdefined, SolverParameters::ReferencePointCalculationStyle::userdefined);
    hvparams->worseReferencePoint = nadir;
    hvparams->betterReferencePoint = ideal;
    
    IQHVSolver hsolver;
    HSSSolver hsssolver;
    HSSParameters* hssparams = new HSSParameters(SolverParameters::ReferencePointCalculationStyle::userdefined, SolverParameters::ReferencePointCalculationStyle::userdefined);
    hssparams->worseReferencePoint = nadir;
    hssparams->betterReferencePoint = ideal;
    hssparams->StoppingSubsetSize = 5;
    auto res = hsssolver.Solve(ds, *hssparams);
    
    double* py_hvcs = new double[16] {
        6.76920703e-05, 1.88971388e-04, 1.93556007e-03, 1.10670045e-02,
            6.69092328e-03, 4.65240351e-05, 9.05157353e-05, 2.38866579e-01,
            2.95675901e-03, 5.64886209e-04, 1.83400107e-04, 1.53125100e-03,
            1.92139151e-03, 2.77026241e-03, 2.10917118e-01, 1.68106613e-03
        };
    auto hv = hsolver.Solve(ds, *hvparams);
    for (int i = 0; i < 16; i++)
    {
        Point p = (*ds)[0];
        ds->remove(0);
        double hvn = hsolver.Solve(ds, *hvparams)->HyperVolume;
        auto hvc = hv->HyperVolume - hvn;
        //std::cout << i << ":" << hv->HyperVolume << " - " << hvn << " = " << hvc << " " << py_hvcs[i] << std::endl;
        std::cout << hv->HyperVolume <<std::endl;
        ds->add(new Point(p));
    }
    auto r = solver.Solve(ds, *params);
    std::cout << "Calculated Minimum Contribution: " << r->MinimumContribution << " ID: " << r->MinimumContributionIndex;
}

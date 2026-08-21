Code samples
=====

.. _tutorials_samples:


Callback usage
-------------------------------------------------------

In order to use callbacks in your code either download a version which support callbacks, or if you are compiling the library directly from sources navigate to the 'include.h' file and set the flag CALLBACKS to a value '1'.
The code below showcases the basic usage of callbacks in C++. Callbacks implementation for python library are to be released soon.

.. code-block:: cpp


    #include <iostream>
    #include <HSSSolver.h>
    #include <SolverParameters.h>

    #include <iomanip>

    void StartCallback(moda::DataSetParameters problemSettings, std::string SolverMessage) {
        std::cout << "Starting: " << SolverMessage << std::endl;
        std::cout << "Problem size:" << problemSettings.nPoints << "x" << problemSettings.NumberOfObjectives << std::endl << std::endl;
    }

    void IterationCallback(int currentIteration, int totalIterations, moda::Result* stepResult) {
        std::cout << "Iteration " << currentIteration << "/" << totalIterations;
        moda::HSSResult* hssResult = (moda::HSSResult*)stepResult;
        std::cout << " Current Hypervolume: "
            << std::fixed << std::setprecision(3) << hssResult->HyperVolume
            << " Last Removed Point Index: " << hssResult->chosenPointIndex << std::endl;
    }

    void EndCallback(moda::DataSetParameters problemSettings, moda::Result* finalResult) {
        moda::HSSResult* hssResult = (moda::HSSResult*)finalResult;
        std::cout << "\nFinished. Hypervolume: "
            << std::fixed << std::setprecision(3) << hssResult->HyperVolume
            << " Subset size: " << hssResult->selectedPoints.size()
            << " Elapsed time: " << finalResult->ElapsedTime / 1000.0 << "s" << std::endl;
    }

    int main()
    {

        moda::DataSet* dataset = new moda::DataSet(2);
        dataset->add(new moda::Point({ 0.1,0.9 })); //0
        dataset->add(new moda::Point({ 0.3,0.7 })); //1
        dataset->add(new moda::Point({ 0.35,0.65 })); //2
        dataset->add(new moda::Point({ 0.4,0.6 })); //3
        dataset->add(new moda::Point({ 0.42,0.58 })); //4
        dataset->add(new moda::Point({ 0.45,0.55 })); //5
        dataset->add(new moda::Point({ 0.6,0.4 })); //6
        dataset->add(new moda::Point({ 0.8,0.2 })); //7
        dataset->add(new moda::Point({ 0.96,0.04 })); //8
        dataset->add(new moda::Point({ 0.99,0.01 })); //9

        moda::HSSSolver solver;
        moda::HSSParameters* params = new moda::HSSParameters();
        params->WorseReferencePointCalculationStyle= moda::SolverParameters::ReferencePointCalculationStyle::zeroone;
        params->StoppingCriteria = moda::HSSParameters::StoppingCriteriaType::SubsetSize;
        params->Strategy = moda::HSSParameters::SubsetSelectionStrategy::Decremental;
        params->StoppingSubsetSize = 5;
        params->CalculateHV = true;

        solver.StartCallback = &StartCallback;
        solver.IterationCallback = &IterationCallback;
        solver.EndCallback = &EndCallback;
        solver.Solve(dataset, *params);
        delete params;
        delete dataset;
    }

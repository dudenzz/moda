// SampleModaProgram.cpp
//

#include <iostream>
#include <sstream>
#include <codecvt>
#include <QEHCSolver.h>
#include <IQHVSolver.h>
#include <DataSet.h>
#include <HSSSolver.h>
#include <DBHVESolver.h>
#include <QHV_BQSolver.h>
#include <QR2.h>
#include <windows.h>
#include <shobjidl.h> 
#include <ostream>
using namespace moda;

struct ExecutionTask {
    int datasetId;
    std::string task;
    DataSet::OptimizationType type;
    SolverParameters::ReferencePointCalculationStyle better;
    SolverParameters::ReferencePointCalculationStyle worse;
    int stopSize;
    int stopTime;
    int iterations;
    QEHCParameters::SearchSubjectOption option;
    HSSParameters::SubsetSelectionStrategy strategy;
    Point* betterUserDefinedPoint;
    Point* worseUserDefinedPoint;

    ExecutionTask(int did, std::string task, DataSet::OptimizationType type, SolverParameters::ReferencePointCalculationStyle better, SolverParameters::ReferencePointCalculationStyle worse,
        int stopSize, int stopTime, QEHCParameters::SearchSubjectOption option, HSSParameters::SubsetSelectionStrategy strategy)
        : datasetId(did), task(task), type(type), better(better), worse(worse), stopSize(stopSize), stopTime(stopTime), option(option), strategy(strategy), iterations(1000) {
    };
    ExecutionTask() {}
};
std::vector<DataSet*> loadedDatasets;
std::vector<ExecutionTask> executionPlan;
HSSSolver hssSolver;
IQHVSolver iqhvSolver;
QEHCSolver qehcSolver;
DBHVESolver dbhveSolver;
QHV_BQSolver bqSolver;
QR2Solver qr2Solver;

HSSParameters* hssParams;
QEHCParameters* qehcParams;
DBHVEParameters* dbhveParams;
IQHVParameters* iqhvParams;
QHV_BQParameters* qhvbqParams;
QR2Parameters* qr2Params;

std::string filePath;
std::string outfilePath = "results.txt";
int menuState = 0;
int datasetId;
int taskId;
bool verbose = false;


void openFolderDialog(int option)
{
    HRESULT hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
    if (SUCCEEDED(hr))
    {
        IFileOpenDialog* pFileOpen;

        hr = CoCreateInstance(CLSID_FileOpenDialog, NULL, CLSCTX_ALL,
            IID_IFileOpenDialog, reinterpret_cast<void**>(&pFileOpen));

        if (SUCCEEDED(hr))
        {
            DWORD dwOptions;
            if (SUCCEEDED(pFileOpen->GetOptions(&dwOptions)))
            {
                pFileOpen->SetOptions(dwOptions | option);
            }
            // ----------------------------------------------------

            hr = pFileOpen->Show(NULL);

            if (SUCCEEDED(hr))
            {
                IShellItem* pItem;
                hr = pFileOpen->GetResult(&pItem);

                if (SUCCEEDED(hr))
                {
                    PWSTR pszFilePath = NULL;
                    hr = pItem->GetDisplayName(SIGDN_FILESYSPATH, &pszFilePath);

                    if (SUCCEEDED(hr))
                    {
                        std::wstring ws(pszFilePath);
                        std::string filePath__(ws.begin(), ws.end());
                        filePath = filePath__;
                        wprintf(L"Selected folder: %s\n", pszFilePath);
                        CoTaskMemFree(pszFilePath);
                    }
                    pItem->Release();
                }
            }
            pFileOpen->Release();
        }
        CoUninitialize();
    }
}
int menu()
{
    int option = 0;
    int tasktype = 1;
    int opttype = 1;
    int stopSize = 0;
    int stopTime = 0;
    int strategy = 0;
    int goalOption = 0;
    int iterations = 0;
    int better = 0;
    int worse = 0;
    Point* betterPoint = new Point();
    Point* worsePoint = new Point();
    int stopstrat = 0;
    std::string taskName;
    DataSet::OptimizationType opt;
    ExecutionTask taskToAdd;
    SolverParameters::ReferencePointCalculationStyle worseStyle;
    SolverParameters::ReferencePointCalculationStyle betterStyle;
    QEHCParameters::SearchSubjectOption searchSubjectOption;
    HSSParameters::SubsetSelectionStrategy subsetSelectionStrategy;
    system("cls");
#pragma region Main Menu
    if (menuState == 0)
    {
        std::cout << "Loaded datasets: " << loadedDatasets.size() << std::endl
            << "Current execution plan tasks: " << executionPlan.size() << std::endl
            << "1. Datasets. " << std::endl
            << "2. Execution plan. " << std::endl
            << "3. Run." << std::endl
            << "4. Settings." << std::endl
            << "5. Exit" << std::endl;
        try {
            std::cin >> option;
        }
        catch (std::exception ex) {
            option = -1;
        }
        switch (option) {
        case 1:
            menuState = 1;
            return 9;
        case 2:
            menuState = 2;
            return 9;
        case 3:
            return 10;
        case 4:
            menuState = 3;
            return 9;
        case 5:
            return 0;

        }
    }
#pragma endregion
#pragma region Datasets
    if (menuState == 1)
    {

        std::cout << "All datasets: " << std::endl;
        for (int iter = 0; iter < loadedDatasets.size(); iter++)
        {
            std::cout << iter << ". " << loadedDatasets[iter]->getParameters()->name << " size: (" << loadedDatasets[iter]->getParameters()->nPoints << "," << loadedDatasets[iter]->getParameters()->NumberOfObjectives << ")" << std::endl;
        }
        std::cout << std::endl << std::endl << "Options" << std::endl;
        std::cout << "1. Add new dataset." << std::endl
            << "2. Remove dataset" << std::endl
            << "3. Add folder of datasets" << std::endl
            << "4. Remove all datasets" << std::endl
            << "5. Print dataset " << std::endl
            << "6. Normalize dataset" << std::endl
            << "7. Back." << std::endl;

        try {
            std::cin >> option;
        }
        catch (std::exception ex) {
            option = -1;
        }
        switch (option) {
        case 1:
            system("cls");
            std::cout << "After you press any button choose a single file to add to the datasets pool. ";
            std::cin.get();
            std::cin.get();
            openFolderDialog(0);
            return 1;
        case 2:
            std::cout << "Which dataset do you want to remove?";
            std::cin >> datasetId;
            return 5;
        case 3:
            std::cout << "After you press any button choose a directory of files to add to the datasets pool. ";
            std::cin.get();
            std::cin.get();
            openFolderDialog(FOS_PICKFOLDERS);
            return 2;
        case 4:
            return 6;
        case 5:
            std::cout << "Which dataset do you want to print?" << std::endl;
            std::cin >> datasetId;
            return 4;
        case 6:
            std::cout << "Which dataset do you want to normalize? (-1 for all datasets)" << std::endl;
            std::cin >> datasetId;
            return 7;
        case 7:
            menuState = 0;
            return 9;

        }
    }
#pragma endregion
#pragma region Tasks
    if (menuState == 2)
    {
        std::cout << "Execution plan: " << std::endl;
        for (int iter = 0; iter < executionPlan.size(); iter++)
        {
            std::string dsName;
            if (executionPlan[iter].datasetId == -1) dsName = "all datasets";
            else dsName = loadedDatasets[executionPlan[iter].datasetId]->getParameters()->name;
            std::cout << iter << ". " << executionPlan[iter].task << "(" << dsName;
            if (executionPlan[iter].task == "QEHC")
            {
                std::string stratName = executionPlan[iter].option == QEHCParameters::SearchSubjectOption::MaximumContribution ? "Maximal Contributor" : "Minimal Contributor";
                std::cout << ",Search Subject=" << stratName;
            }
            if (executionPlan[iter].task == "HSS")
            {
                std::string stratName = executionPlan[iter].strategy == HSSParameters::SubsetSelectionStrategy::Incremental ? "Incremental" : "Decremental";
                std::cout << ",Strategy=" << stratName;
            }
            if (executionPlan[iter].task == "DBHVE")
            {

                std::cout << ",MaxTime=" << executionPlan[iter].stopTime;
            }
            std::string typeName = executionPlan[iter].type == DataSet::OptimizationType::maximization ? "Maximization" : "Minimization";
            std::cout << ",Type=" << typeName << ")" << std::endl;
        }
        std::cout << std::endl << std::endl << "Options" << std::endl;
        std::cout << "1. Add new task." << std::endl
            << "2. Remove task" << std::endl
            << "3. Clear tasks" << std::endl
            << "4. Back." << std::endl;
        try {
            std::cin >> option;
        }
        catch (std::exception ex) {
            option = -1;
        }
        switch (option) {
        case 1:


            std::cout << "Which dataset do you want to make a task for?(-1 for all datasets)" << std::endl;
            std::cin >> datasetId;


            std::cout << "Task type:" << std::endl;
            std::cout << "1. IQHV" << std::endl;
            std::cout << "2. DBHVE" << std::endl;
            std::cout << "3. HSS" << std::endl;
            std::cout << "4. QEHC" << std::endl;
            std::cout << "5. QHV_BQ" << std::endl;
            std::cout << "6. R2" << std::endl;
            try {
                std::cin >> tasktype;
            }
            catch (std::exception ex) {
                return 9;
            }

            std::cout << "Optimization type:" << std::endl;
            std::cout << "1. Maxmization" << std::endl;
            std::cout << "2. Minimization" << std::endl;
            try {
                std::cin >> opttype;
            }
            catch (std::exception ex) {
                return 9;
            }



            std::cout << "Better Reference Point" << std::endl;
            std::cout << "1. All ones(maximization) or zeroes(minimization)" << std::endl;
            std::cout << "2. Exact values" << std::endl;
            std::cout << "3. Exact values +- 10% of exact values" << std::endl;
            std::cout << "4. Exact values +- 0.001 of exact values" << std::endl;
            std::cout << "5. All elevens(maximization) or negative elevens(minimization)" << std::endl;
            std::cout << "6. My own definition (applicable only to a specific dataset)" << std::endl;

            try {
                std::cin >> better;
            }
            catch (std::exception ex) {
                return 9;
            }
            if (better == 6)
            {
                if (datasetId == -1)
                {
                    std::cout << "You can't use user defined reference points for multiple datasets. Potential dimensionality mismatch.";
                    return 9;
                }
                std::cout << "Better reference point: [";
                betterPoint = new Point(loadedDatasets[datasetId]->getParameters()->NumberOfObjectives);
                for (int i = 0; i < loadedDatasets[datasetId]->getParameters()->NumberOfObjectives; i++)
                {
                    std::cin >> betterPoint->ObjectiveValues[i];
                    std::cout << ",";

                }
                std::cout << "]";
            }

            std::cout << "Worse Reference Point" << std::endl;
            std::cout << "1. All ones(minimization) or zeroes(maximization)" << std::endl;
            std::cout << "2. Exact values" << std::endl;
            std::cout << "3. Exact values +- 10% of exact values" << std::endl;
            std::cout << "4. Exact values +- 0.001 of exact values" << std::endl;
            std::cout << "5. All elevens(minimization) or negative elevens(maximization)" << std::endl;
            std::cout << "6. My own definition (applicable only to a specific dataset)" << std::endl;

            try {
                std::cin >> worse;
            }
            catch (std::exception ex) {
                return 9;
            }
            if (worse == 6)
            {
                if (datasetId == -1)
                {
                    std::cout << "You can't use user defined reference points for multiple datasets. Potential dimensionality mismatch.";
                    return 9;
                }
                std::cout << "Worse reference point: [";
                worsePoint = new Point(loadedDatasets[datasetId]->getParameters()->NumberOfObjectives);
                for (int i = 0; i < loadedDatasets[datasetId]->getParameters()->NumberOfObjectives; i++)
                {
                    std::cin >> worsePoint->ObjectiveValues[i];
                    std::cout << ",";

                }
                std::cout << "]";
            }
            if (tasktype == 2)
            {

                std::cout << "Max MC Iterations is set to 10e6\nMax Execution time (ms):";
                try {
                    std::cin >> stopTime;
                }
                catch (std::exception ex) {
                    return 9;
                }
            }
            if (tasktype == 3)
            {
                int temp;
                std::cout << "When to stop:" << std::endl;
                std::cout << "1. Subset size" << std::endl;
                std::cout << "2. Execution time" << std::endl;
                try {
                    std::cin >> stopstrat;
                }
                catch (std::exception ex) {
                    return 9;
                }
                if (stopstrat == 1)
                {

                    std::cout << "Subset size:";
                    try {
                        std::cin >> stopSize;
                    }
                    catch (std::exception ex) {
                        return 9;
                    }
                }
                if (stopstrat == 2)
                {

                    std::cout << "Execution time (ms):";
                    try {
                        std::cin >> stopTime;
                    }
                    catch (std::exception ex) {
                        return 9;
                    }
                }
                std::cout << "Strategy:" << std::endl;
                std::cout << "1. Incremental" << std::endl;
                std::cout << "2. Decremental" << std::endl;
                try {
                    std::cin >> strategy;
                }
                catch (std::exception ex) {
                    return 9;
                }
            }

            if (tasktype == 4)
            {

                std::cout << "What is being looked for:" << std::endl;
                std::cout << "1. Maximum Contributor" << std::endl;
                std::cout << "2. Minimum Contributor" << std::endl;
                try {
                    std::cin >> goalOption;
                }
                catch (std::exception ex) {
                    return 9;
                }
            }
            if (tasktype == 5)
            {

                std::cout << "Max MC Iterations is set to 10e6\nMax Execution time (ms):";
                try {
                    std::cin >> stopTime;
                }
                catch (std::exception ex) {
                    return 9;
                }
            }

            switch (tasktype)
            {
            case 1: taskName = "IQHV"; break;
            case 2: taskName = "DBHVE"; break;
            case 3: taskName = "HSS"; break;
            case 4: taskName = "QEHC"; break;
            case 5: taskName = "QHV_BQ"; break;
            case 6: taskName = "R2"; break;
            }
            opt = opttype == 1 ? DataSet::OptimizationType::maximization : DataSet::OptimizationType::minimization;
            switch (better) {
            case 1: betterStyle = SolverParameters::ReferencePointCalculationStyle::zeroone; break;
            case 2: betterStyle = SolverParameters::ReferencePointCalculationStyle::exact; break;
            case 3: betterStyle = SolverParameters::ReferencePointCalculationStyle::tenpercent; break;
            case 4: betterStyle = SolverParameters::ReferencePointCalculationStyle::epsilon; break;
            case 5: betterStyle = SolverParameters::ReferencePointCalculationStyle::pymoo; break;
            case 6: betterStyle = SolverParameters::ReferencePointCalculationStyle::userdefined; break;
            default: betterStyle = SolverParameters::ReferencePointCalculationStyle::exact; break;
            }
            switch (worse) {
            case 1: worseStyle = SolverParameters::ReferencePointCalculationStyle::zeroone; break;
            case 2: worseStyle = SolverParameters::ReferencePointCalculationStyle::exact; break;
            case 3: worseStyle = SolverParameters::ReferencePointCalculationStyle::tenpercent; break;
            case 4: worseStyle = SolverParameters::ReferencePointCalculationStyle::epsilon; break;
            case 5: worseStyle = SolverParameters::ReferencePointCalculationStyle::pymoo; break;
            case 6: worseStyle = SolverParameters::ReferencePointCalculationStyle::userdefined; break;
            default: worseStyle = SolverParameters::ReferencePointCalculationStyle::exact; break;
            }
            searchSubjectOption = goalOption == 1 ? QEHCParameters::SearchSubjectOption::MaximumContribution : QEHCParameters::SearchSubjectOption::MinimumContribution;
            subsetSelectionStrategy = strategy == 1 ? HSSParameters::SubsetSelectionStrategy::Incremental : HSSParameters::SubsetSelectionStrategy::Decremental;
            taskToAdd = ExecutionTask(datasetId, taskName, opt, betterStyle, worseStyle, stopSize, stopTime, searchSubjectOption, subsetSelectionStrategy);
            taskToAdd.betterUserDefinedPoint = betterPoint;
            taskToAdd.worseUserDefinedPoint = worsePoint;
            executionPlan.push_back(taskToAdd);
            std::cout << "Task added to the plan. Press any key to continue.";
            std::cin.get();
            return 3;
        case 2:
            std::cout << "Which task do you want to delete?" << std::endl;
            std::cin >> taskId;
            return 11;
        case 3:
            return 12;
        case 4:
            menuState = 0;
            return 9;

        }
    }
#pragma endregion
#pragma region Settings
    if (menuState == 3)
    {
        std::string verbosity_str = verbose ? "True" : "False";
        std::cout << "Verbosity: " << verbosity_str << std::endl;
        std::cout << "Results file: " << outfilePath << std::endl;
        std::cout << std::endl << std::endl << "Options" << std::endl;
        std::cout << "1. Toggle verbosity. " << std::endl
            << "2. Change result file path. " << std::endl
            << "3. Back. " << std::endl;

        try {
            std::cin >> option;
        }
        catch (std::exception ex) {
            option = -1;
        }
        switch (option) {
        case 1:
            verbose = !verbose;
            return 9;
        case 2:
            std::cout << "New output file path: ";
            std::cin >> outfilePath;
            return 9;
        case 3:
            menuState = 0;
            return 9;
        }
    }
#pragma endregion
    return 9;



}
//execution codes:
// 0 - Exit program
// 1 - Add a single file to the dataset list
// 2 - Add a directory of files to the dataset list
// 3 - Add a new task to execution plan
// 4 - Print dataset
// 5 - Remove dataset;
// 6 - Clear datasets;
// 7 - Normalize dataset;
// 8 - Remove task from execution plan
// 9 - No action;
// 10 - Execute plan;
// 11 - Remove task;
// 12 - Clear all tasks;
int main()
{
    std::ofstream outputFile;
    int taskIter = 0;

	//Sample seed dataset and task for debugging purposes. Comment out if not needed.
    //DataSet* DS = DataSet::LoadFromFilename("C://debugging/debug.txt");
    //loadedDatasets.push_back(DS);
     
	//Sample seed parameters and task for debugging purposes. Comment out if not needed.
    //executionPlan.push_back(ExecutionTask(0, "HSS", DataSet::OptimizationType::minimization, SolverParameters::ReferencePointCalculationStyle::pymoo, SolverParameters::ReferencePointCalculationStyle::pymoo, 100, 0, QEHCParameters::SearchSubjectOption::Both, HSSParameters::SubsetSelectionStrategy::Decremental));


    std::vector<DataSet*> datasets;
    DataSet* ds;
    int executionCode = -1;
    while (executionCode != 0) {
        executionCode = menu();
        switch (executionCode) {
        case 1:
            ds = DataSet::LoadFromFilename(filePath);
            loadedDatasets.push_back(ds);
            std::cout << "Press any button to continue." << std::endl;
            break;
        case 2:
            datasets = DataSet::LoadBulk(filePath);
            for (auto ds : datasets)
            {
                loadedDatasets.push_back(ds);
            }
            std::cout << "Press any button to continue." << std::endl;
            break;

        case 4:
            try {
                for (auto p : loadedDatasets[datasetId]->points)
                {
                    for (int i = 0; i < loadedDatasets[datasetId]->getParameters()->NumberOfObjectives; i++)
                    {
                        std::cout << p->ObjectiveValues[i] << " ";
                    }
                    std::cout << std::endl;
                }
                std::cout << "Press any button to continue." << std::endl;
                std::cin.get();
            }
            catch (std::exception ex)
            {
                std::cout << "Couldn't print dataset (" << ex.what() << ") Press any button to continue." << std::endl;
                std::cin.get();
                std::cin.get();
            }
            break;
        case 5:
            try {
                loadedDatasets.erase(loadedDatasets.begin() + datasetId);
            }
            catch (std::exception ex)
            {
                std::cout << "Couldn't erase dataset (" << ex.what() << ") Press any button to continue." << std::endl;
                std::cin.get();
                std::cin.get();
            }
            break;
        case 6:
            loadedDatasets.clear();
            break;
        case 7:
            if (datasetId == -1)
                for (auto ds : loadedDatasets) ds->normalize();
            else
                loadedDatasets[datasetId]->normalize();
            break;
        case 9:
            break;
        case 10:
            outputFile = std::ofstream(outfilePath);

            for (auto task : executionPlan)
            {
                taskIter += 1;
                std::vector<int> datasetIds;
                if (task.datasetId == -1)
                {
                    for (int i = 0; i < loadedDatasets.size(); i++) datasetIds.push_back(i);
                }
                else datasetIds.push_back(task.datasetId);
                for (int currentId : datasetIds)
                {
                    DataSet* ds = loadedDatasets[currentId];
                    if (!verbose)
                        std::cout << "Starting task " << task.task << "(" << ds->getParameters()->name << ")" << std::endl;
                    if (verbose)
                        std::cout << "Running: " << task.task << "(" << ds->getParameters()->name << ") Task:" << taskIter << "/" << executionPlan.size() << " Dataset: " << currentId << "/" << datasetIds.size() << "                                                             \r";
                    outputFile << task.task << "\t" << ds->getParameters()->name << "\t";
                    if (task.type == DataSet::OptimizationType::maximization)
                        outputFile << "Maximization\t";
                    else
                        outputFile << "Minimization\t";
                    if ("IQHV" == task.task)
                    {
                        iqhvParams = new IQHVParameters(task.worse, task.better);
                        ds->typeOfOptimization = task.type;
                        iqhvParams->betterReferencePoint = task.betterUserDefinedPoint;
                        iqhvParams->worseReferencePoint = task.worseUserDefinedPoint;
                        auto result = iqhvSolver.Solve(ds, *iqhvParams);
                        if (!verbose)
                            std::cout << "Task finished. Result: " << result->HyperVolume << " Elapsed: " << result->ElapsedTime << "ms" << std::endl;
                        outputFile << result->ElapsedTime << "\t" << result->HyperVolume << "\t\n";

                    }
                    if ("QEHC" == task.task)
                    {

                        qehcParams = new QEHCParameters(task.worse, task.better);
                        qehcParams->SearchSubject = task.option;
                        qehcParams->betterReferencePoint = task.betterUserDefinedPoint;
                        qehcParams->worseReferencePoint = task.worseUserDefinedPoint;
                        ds->typeOfOptimization = task.type;
                        auto result = qehcSolver.Solve(ds, *qehcParams);
                        if (task.option == QEHCParameters::SearchSubjectOption::MinimumContribution && !verbose) {
                            std::cout << "Task finished. Minimum contribution: " << result->MinimumContribution << " Minimum contribution id: " << result->MinimumContributionIndex << " Elapsed: " << result->ElapsedTime << "ms" << std::endl;
                            outputFile << result->ElapsedTime << "\t" << result->MinimumContribution << "\t" << result->MinimumContributionIndex << "\n";
                        }
                        if (task.option == QEHCParameters::SearchSubjectOption::MaximumContribution && !verbose)
                        {
                            std::cout << "Task finished. Maximum contribution:: " << result->MaximumContribution << " Maximum contribution id: " << result->MaximumContributionIndex << " Elapsed: " << result->ElapsedTime << "ms" << std::endl;
                            outputFile << result->ElapsedTime << "\t" << result->MaximumContribution << "\t" << result->MaximumContributionIndex << "\n";
                        }
                    }
                    if ("HSS" == task.task)
                    {

                        hssParams = new HSSParameters(task.worse, task.better);
                        hssParams->betterReferencePoint = task.betterUserDefinedPoint;
                        hssParams->worseReferencePoint = task.worseUserDefinedPoint;
                        if (task.stopTime != 0 && task.stopSize == 0)
                        {
                            hssParams->StoppingCriteria = HSSParameters::StoppingCriteriaType::Time;
                            hssParams->StoppingTime = task.stopTime;
                        }
                        else if (task.stopTime == 0 && task.stopSize != 0)
                        {
                            hssParams->StoppingCriteria = HSSParameters::StoppingCriteriaType::SubsetSize;
                            hssParams->StoppingSubsetSize = task.stopSize;
                        }
                        else {
                            if (!verbose)
                                std::cout << "Choose either stop size or time. Setting default stop size of 1" << std::endl;
                            hssParams->StoppingCriteria = HSSParameters::StoppingCriteriaType::SubsetSize;
                            hssParams->StoppingSubsetSize = 1;
                        }

                        ds->typeOfOptimization = task.type;
                        for (int i = 0; i < 10000000; i++) hssSolver.Solve(ds, *hssParams);
                        auto result = hssSolver.Solve(ds, *hssParams);
                        outputFile << result->ElapsedTime << "\t";
                        if (!verbose)

                            std::cout << "Task finished. Selected points: (";
                        for (auto point : result->selectedPoints)
                        {
                            if (!verbose)
                                std::cout << point << ",";
                            outputFile << point << ",";
                        }
                        if (!verbose)
                            std::cout << ") Elapsed: " << result->ElapsedTime << "ms" << std::endl;
                        outputFile << "\t\n";

                    }
                    if ("DBHVE" == task.task)
                    {
                        dbhveParams = new DBHVEParameters(task.worse, task.better);
                        dbhveParams->betterReferencePoint = task.betterUserDefinedPoint;
                        dbhveParams->worseReferencePoint = task.worseUserDefinedPoint;
                        dbhveParams->MCiterations = 10e6;
                        dbhveParams->MaxEstimationTime = task.stopTime;
                        ds->typeOfOptimization = task.type;
                        auto result = dbhveSolver.Solve(ds, *dbhveParams);
                        if (!verbose)
                            std::cout << "Task finished. Result: " << result->HyperVolumeEstimation << " Elapsed: " << result->ElapsedTime << "ms" << std::endl;
                        outputFile << result->ElapsedTime << "\t" << result->HyperVolumeEstimation << "\t\n";
                    }
                    if ("QHV_BQ" == task.task)
                    {
                        qhvbqParams = new QHV_BQParameters(task.worse, task.better);
                        qhvbqParams->betterReferencePoint = task.betterUserDefinedPoint;
                        qhvbqParams->worseReferencePoint = task.worseUserDefinedPoint;
                        qhvbqParams->MaxEstimationTime = task.stopTime;
                        ds->typeOfOptimization = task.type;
                        auto result = bqSolver.Solve(ds, *qhvbqParams);
                        if (!verbose)
                            std::cout << "Task finished. Result: " << result->HyperVolumeEstimation << " Elapsed: " << result->ElapsedTime << "ms" << std::endl;
                        outputFile << result->ElapsedTime << "\t" << result->HyperVolumeEstimation << "\t\n";
                    }
                    if ("R2" == task.task)
                    {
                        qr2Params = new QR2Parameters(task.worse, task.better);
                        qr2Params->betterReferencePoint = task.betterUserDefinedPoint;
                        qr2Params->worseReferencePoint = task.worseUserDefinedPoint;
                        ds->typeOfOptimization = task.type;
                        auto result = qr2Solver.Solve(ds, *qr2Params);
                        if (!verbose)
                            std::cout << "Task finished. Result: " << result->R2 << " Elapsed: " << result->ElapsedTime << "ms" << std::endl;
                        outputFile << result->ElapsedTime << "\t" << result->R2 << "\t\n";

                    }
                    if (!verbose)
                        std::cout << std::endl;
                }

            }
            if (verbose) std::cout << std::endl;
            outputFile.close();
            std::cout << "Press any button to continue." << std::endl;
            std::cin.get();
            break;
        case 11:
            executionPlan.erase(executionPlan.begin() + taskId);
            break;
        case 12:
            executionPlan.clear();
        }

        std::cin.clear();
        std::cin.ignore(1000, '\n');

    }

}

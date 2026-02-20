#include "Helpers.h"
namespace moda {
    bool progressCalledAfterStart = false;
    int iter = 0;

    void ProgressBarCallback(int currentIteration, int totalIterations, Result* stepResult)
    {
        progressCalledAfterStart = true;
        std::ostringstream oss, lbss, ubss, miss, mxss;
        std::string result, lb, ub, minimum, maximum;
        std::string message;
        BoundedResult* stepResultMasked;
        QEHCResult* stepCntrResultMasked;
        HSSResult* stepSSResultMasked;
        switch (stepResult->type)
        {
        case Result::Hypervolume: 
            
            oss << std::setprecision(5) << ((HypervolumeResult*)stepResult)->HyperVolume;
            result = oss.str();
            message = std::to_string(stepResult->ElapsedTime / 1000.0) + "s" + " hv=" + result;
            make_progress(currentIteration / (float)totalIterations, currentIteration, totalIterations, message = message);
            break;

        case Result::R2:

            oss << std::setprecision(5) << ((R2Result*)stepResult)->R2;
            result = oss.str();
            message = std::to_string(stepResult->ElapsedTime / 1000.0) + "s" + " r2=" + result;
            make_progress(currentIteration / (float)totalIterations, currentIteration, totalIterations, message = message);
            break;
        case Result::Estimation:
            stepResultMasked = (BoundedResult*)stepResult;
            oss << std::setprecision(4) << stepResultMasked->HyperVolumeEstimation;
            result = oss.str();
            

            lbss << std::setprecision(4) << stepResultMasked->LowerBound;
            lb = lbss.str();

            ubss << std::setprecision(4) << stepResultMasked->UpperBound;
            ub = ubss.str();
            message = std::to_string(stepResult->ElapsedTime / 1000.0) + "s" + " hv=" + result + " lb=" + lb + " ub=" + ub;
            make_progress(currentIteration / (float)totalIterations, currentIteration, totalIterations, message = message);
            break;
        case Result::Contribution:
            stepCntrResultMasked = (QEHCResult*)stepResult;
            miss << std::setprecision(3) << stepCntrResultMasked->MinimumContribution;
            minimum = miss.str();
            mxss << std::setprecision(3) << stepCntrResultMasked->MaximumContribution;
            maximum = mxss.str();

            message = std::to_string(stepResult->ElapsedTime / 1000.0) + "s" + " max=" + maximum + " min=" + minimum;
            make_progress(currentIteration / (float)totalIterations, currentIteration, totalIterations, message = message);
            break;

        case Result::SubsetSelection:
            stepSSResultMasked = (HSSResult*)stepResult;
            
            miss << std::setprecision(3) << stepSSResultMasked->HyperVolume;
            result = miss.str();
            message = std::to_string(stepResult->ElapsedTime / 1000.0) + "s" + " hv=" + result + " size=" + std::to_string(stepSSResultMasked->selectedPoints.size());
            make_progress(currentIteration / (float)totalIterations, currentIteration, totalIterations, message = message);
            break;

;
        }
    }

    void DefaultStartCallback(DataSetParameters problemSettings, std::string SolverMessage)
    {
        iter = 0;
        progressCalledAfterStart = false;
        std::cout << SolverMessage <<std::endl<<"\tIteration: "  << problemSettings.sampleNumber << std::endl << "\tData file name: " << problemSettings.filename << std::endl;
    }
    void FileSaveEndCallback(DataSetParameters problemSettings, Result* finalResult)
    {
        if (progressCalledAfterStart)
            std::cout << std::endl;
        HSSResult* ssr;
        switch (finalResult->type)
        {
        case Result::SubsetSelection:
            ssr = ((HSSResult*)finalResult);
            std::cout << "\tVolume Value: " << ssr->HyperVolume << std::endl << "\tSubset Size: " << ssr->selectedPoints.size() << std::endl << "\tElapsed time: " << finalResult->ElapsedTime / 1000.0 << "s" << std::endl << std::endl;
            std::ofstream outputFile("C:/Users/kubad/Documents/HV_result/result.csv", std::ios_base::app);
            std::string sep = "\t";
            if (ssr->methodName == "HSS Dec experimental")
                outputFile << problemSettings.name << sep << problemSettings.sampleNumber << sep << problemSettings.nPoints << sep << problemSettings.NumberOfObjectives << sep << ssr->HyperVolume << sep << ssr->ElapsedTime;
            else
                outputFile << sep << ssr->HyperVolume << sep << ssr->ElapsedTime << std::endl;
            outputFile.close();
            break;
        }
    }
    void DefaultEndCallback(DataSetParameters problemSettings, Result* finalResult)
    {
        if (progressCalledAfterStart)
            std::cout << std::endl;
        HypervolumeResult* hvr;
        R2Result* r2r;
        BoundedResult* bvr;
        QEHCResult* cvr;
        HSSResult* ssr;
        switch (finalResult->type)
        {
        case Result::Hypervolume:

            hvr = ((HypervolumeResult*)finalResult);
            std::cout << "\tVolume Value: " << hvr->HyperVolume << std::endl << "\tElapsed time: "  << finalResult->ElapsedTime / 1000.0 << "s" << std::endl << std::endl;
            break;
        case Result::R2:

            r2r = ((R2Result*)finalResult);
            std::cout << "\tR2 Value: " << r2r->R2 << std::endl;

            #ifdef CALCULATE_R2_WITH_HV
            std::cout << "\tHV Value: " << r2r->Hypervolume << std::endl;
            #endif // CALCULATE_R2_WITH_HV

            std::cout <<"\tElapsed time: " << finalResult->ElapsedTime / 1000.0 << "s" << std::endl << std::endl;
            break;
        case Result::Estimation:
            bvr = ((BoundedResult*)finalResult);
            std::cout << "\tVolume Value: " << bvr->HyperVolumeEstimation << std::endl << "\tElapsed time: " << finalResult->ElapsedTime / 1000.0 << "s" << std::endl << std::endl;

            break;
        case Result::Contribution:
            cvr = ((QEHCResult*)finalResult);
            std::cout << "\tMaximum Volume: " << cvr->MaximumContribution << std::endl
                << "\tMinimum Volume: " << cvr->MinimumContribution << std::endl
                << "\tMaximum Volume Index: " << cvr->MaximumContributionIndex << std::endl
                << "\tMinimum Volume Index: " << cvr->MinimumContributionIndex << std::endl
                << "\tElapsed time: " << finalResult->ElapsedTime / 1000.0 << "s" << std::endl << std::endl;

            break;
        case Result::SubsetSelection:
            ssr = ((HSSResult*)finalResult);
            std::cout << "\tVolume Value: " << ssr->HyperVolume << std::endl << "\tSubset Size: " << ssr->selectedPoints.size() << std::endl << "\tElapsed time: " << finalResult->ElapsedTime / 1000.0 << "s" << std::endl << std::endl;
            
            std::ofstream file("C:/Hypervolume/HVE/HVE/source-code/data/data_prep/result_test", std::ios_base::app);
            file << problemSettings.nPoints << "\t" << problemSettings.filename << "\t" << problemSettings.NumberOfObjectives << "\t" << problemSettings.name << "\t" << ssr->methodName << "\t" << ssr->HyperVolume << "\t" << ssr->ElapsedTime << "\t" << ssr->selectedPoints.size() << std::endl;
            file.close();
            break;
        }
    }

    std::vector<std::string> split(std::string& s, const std::string& delimiter)
    {
        std::vector<std::string> tokens;
        int pos = 0;
        std::string token;
        while ((pos = s.find(delimiter)) != std::string::npos)
        {
            token = s.substr(0, pos);
            tokens.push_back(token);

            s = s.erase(0, pos + delimiter.length());
        }
        tokens.push_back(s);

        return tokens;
    }

    std::string join(const std::vector<std::string>& lst, const std::string& delimiter)
    {
        std::string ret;
        for (const auto& s : lst)
        {
            if (!ret.empty())
                ret += delimiter;
            ret += s;
        }
        return ret;
    }

    std::tuple<int, int> parseSettings(const std::string settingsString)
    {
        int state = 0;
        std::string dim_string = "";
        std::string nPoints_string = "";
        for (char letter : settingsString)
        {
            if (letter == 'd')
                continue;
            if (letter == 'n')
            {
                state = 1;
                continue;
            }
            if (state == 0)
                dim_string += letter;
            if (state == 1)
                nPoints_string += letter;
        }
        return std::make_tuple(stoi(dim_string), stoi(nPoints_string));
    }

    void make_progress(float progress,int n1, int n2, std::string message)
    {

        int barWidth = 30;

        std::cout << "\t[";
        int pos = barWidth * progress;
        for (int i = 0; i < barWidth; ++i)
        {
            if (i < pos)
                std::cout << "=";
            else if (i == pos)
                std::cout << ">";
            else
                std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << " % (" << n1 << "/" << n2 << ") " << message << "\r";
        std::cout.flush();
        std::cout.flush();
    }

    int countFilesInDirectory(std::string directory)
    {

#ifdef _MSC_VER 
        auto dirIter = std::filesystem::directory_iterator(directory);
#else
        auto dirIter = std::filesystem::directory_iterator(directory);
#endif // _MSC_VER  



        return std::count_if(
            begin(dirIter),
            end(dirIter),
            [](auto& entry) { return entry.is_regular_file(); }
        );

    }
    DType logarithm(DType param, DType base)
    {
        return log(param) / log(base);
    }

}
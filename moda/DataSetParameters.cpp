#include "DataSetParameters.h"
#include <regex>
namespace moda {
    DataSetParameters::DataSetParameters() {
        NumberOfObjectives = 0;
        sampleNumber = -1;
        nPoints = 0;
        name = "default name";
    };

    DataSetParameters::DataSetParameters(std::string name, int NumberOfObjectives, int nPoints, int sampleNumber) : name(name), NumberOfObjectives(NumberOfObjectives), sampleNumber(sampleNumber)
    {
        filename = "In Memory Problem";
    };
    DataSetParameters::~DataSetParameters()
    {

    }
    DataSetParameters::DataSetParameters(std::string filename)
    {
        this->filename = filename;
        std::vector<std::string> tokens_full = split(filename, "\\");
        filename = tokens_full[tokens_full.size() - 1];
        std::regex pattern1("^[a-z0-9_]+_d[0-9]+n[0-9]+_[0-9]+$");
        //data_4_100_concave_triangular_1
        std::regex pattern2("^[a-z0-9_]+_[0-9]+_[0-9]+_[_a-z0-9]+_[0-9]+$");

        if (std::regex_match(filename, pattern1))
        {

            std::vector<std::string> tokens = split(filename, "_");
            auto n_tokens = tokens.size();
            sampleNumber = stoi(tokens[n_tokens - 1]);
            std::string toParse = tokens[n_tokens - 2];
            auto settings = parseSettings(toParse);
            NumberOfObjectives = std::get<0>(settings);
            nPoints = std::get<1>(settings);
            std::vector<std::string> name_tokens(tokens.begin(), tokens.end() - 2);
            name = join(name_tokens, "_");
        }
        if (std::regex_match(filename, pattern2))
        {
            std::vector<std::string> tokens = split(filename, "_");
            auto n_tokens = tokens.size();

            sampleNumber = stoi(tokens[n_tokens - 1]);
          
            nPoints = stoi(tokens[2]);
            NumberOfObjectives = stoi(tokens[1]);
            std::vector<std::string> name_tokens(tokens.begin() + 3, tokens.end() - 1);
            name = join(name_tokens, "_");
        }
    };
}

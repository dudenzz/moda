#include "include.h"
#include "Helpers.h"
#ifndef C_DATASET_SETTINGS
#define C_DATASET_SETTINGS
namespace moda {
    class DataSetParameters {
    public:
#pragma region unused
        enum OptimizationType { max, min };
        OptimizationType optType;
#pragma endregion
#pragma region Dataset properties
        /// <summary>
        /// filename
        /// </summary>
        std::string filename;
        /// <summary>
        /// name of the experiment
        /// </summary>
        std::string name;
        /// <summary>
        /// number of objectives
        /// </summary>
        int NumberOfObjectives;
        /// <summary>
        /// number of points in the dataset
        /// </summary>
        int nPoints;
        /// <summary>
        /// number of sample in the experiment
        /// </summary>
        int sampleNumber;
#pragma endregion
#pragma region constructors
        /// <summary>
        /// non parametrized constructor
        /// </summary>
        DataSetParameters();
        /// <summary>
        /// default constructor
        /// </summary>
        /// <param name="name">name of the experiment</param>
        /// <param name="dimensions">number of objectives</param>
        /// <param name="nPoints">number of points in the file</param>
        /// <param name="sampleNumber">sample number in the experiment</param>
        DataSetParameters(std::string name, int dimensions, int nPoints, int sampleNumber);
        /// <summary>
        /// This constructor parses the filename and gathers the dataset properties from the filename
        /// </summary>
        /// <param name="filename">filename</param>
        DataSetParameters(std::string filename);
        ~DataSetParameters();
#pragma endregion
    };
}

#endif //C_DATASET_SETTINGS
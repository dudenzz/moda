#pragma once
#include "include.h"
#include "vector"
namespace moda {
   
    class Result {
    public:
        /// <summary>
        ///  Time passed since the begining of the calculation given in ms
        /// </summary>
        int ElapsedTime;
        /// <summary>
        /// Is this the final result of the calculation
        /// </summary>
        bool FinalResult;
        enum ResultType { Hypervolume, Estimation, Contribution, SubsetSelection, R2 };
        ResultType type;
        std::string methodName;
        Result();


    };
    
    class HypervolumeResult : public Result {
    public:
        /// <summary>
        /// Calculated Hyper HyperVolume
        /// </summary> 
        float HyperVolume;
    };
    class R2Result : public Result {
    public:
        /// <summary>
        /// Calculated Hyper HyperVolume
        /// </summary> 
        float R2;
        float Hypervolume;
    };

    //TODO: wywo�ywanie z callbackami/bez to nie to samo co liczenie qhv w ka�dym stepie
    class HSSResult : public Result {
    public:
        /// <summary>
        /// Selected set of points with 
        /// </summary> 
        std::vector<int> selectedPoints;
        int chosenPointIndex;
        float HyperVolume;

        
    };

    class BoundedResult : public Result {
    public:
        /// <summary>
        /// Ta flaga oznacza, �e lower i upper bound s� dok�adne. Mamy pewno��, �e dok�adna warto�c jest pomi�dzy lower i upper bound
        /// Przy metodze Monte Carlo, tak nie jest.
        /// </summary>
        bool Guaranteed;
        /// <summary>
        /// Lower bound of the Hyper HyperVolume
        /// </summary>
        float LowerBound;
        /// <summary>
        /// Upper bound of the Hyper HyperVolume
        /// </summary>
        float UpperBound;
        /// <summary>
        /// Estimation of the Hyper HyperVolume
        /// </summary>
        float HyperVolumeEstimation;
    };
    class QEHCResult : public Result {
    public:
        /// <summary>
        /// Denotes the lowest contribution value for any single point in the set. 
        /// </summary>
        float MaximumContribution;
        /// <summary>
        /// Denotes the highest HV contribution value for any single point in the set. 
        /// </summary>
        float MinimumContribution;
        /// <summary>
        /// Denotes the index of a point with the highest HV contribution value for any single point in the set. 
        /// </summary>
        int MaximumContributionIndex;
        /// <summary>
        /// Denotes the index of a point with the lowest HV contribution value for any single point in the set. 
        /// </summary>
        int MinimumContributionIndex;
    };
    class QHV_BQResult : public BoundedResult {

    };
    class QHV_BRResult : public BoundedResult {

    };
    class MCHVResult : public BoundedResult {

    };

    class DBHVEResult : public BoundedResult {

    };

}
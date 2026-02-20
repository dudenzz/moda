

#ifndef C_SOLVER_SETTINGS
#define C_SOLVER_SETTINGS
#include "DataSet.h"
namespace moda {
    class SwitchParameters {
    public:
        /// <summary>
        /// time in ms after which BQHV switches to Monte Carlo
        /// </summary>
        int switchTime;
        /// <summary>
        /// minimum gap between upper and lower reference point in estimation methods, if reached the estimation stops
        /// </summary>
        DType gap;
        /// <summary>
        /// maximum problem stack size for Monte Carlo based methods
        /// </summary>
        int maxStackProblemSize;
        /// <summary>
        /// maximum nubmer of iterations for the BQHV before switch to MCHV
        /// </summary>
        int iterations;

        static SwitchParameters defaultSettings()
        {
            SwitchParameters dSett;
            dSett.switchTime = 2000;
            dSett.gap = 0;
            dSett.maxStackProblemSize = 9000000;
            dSett.iterations = 10000000;
            return dSett;
        }
    };
    class SolverParameters {
    public:
        /// <summary>
        /// types of worse reference points estimation; gap: worse = nadir - epislon; tenpercent: worse = worse - 0.1(better-worse); zeroone: worse = {0,0,0...,0}; userdefined: worse = undefined; exact: worse = nadir
        /// </summary>
        enum ReferencePointCalculationStyle {epsilon, tenpercent, zeroone, userdefined, exact , pymoo};
        
        ReferencePointCalculationStyle BetterReferencePointCalculationStyle;
        ReferencePointCalculationStyle WorseReferencePointCalculationStyle;
        /// <summary>
        /// should the solver use iteration callbacks
        /// </summary>
        bool callbacks;
        /// <summary>
        /// for minimization: lower boundary point of the problem; for maximization: upper boundary point of the problem
        /// </summary>
        Point* GetWorseReferencePoint(DataSet *set);
        Point* GetBetterReferencePoint(DataSet *set);
        /// <summary>
        /// User defined reference point;for minimization: lower boundary point of the problem; for maximization: upper boundary point of the problem
        /// </summary>
        Point* worseReferencePoint;
        /// <summary>
        /// User defined reference point; for minimization: upper boundary point of the problem; for maximization: lower boundary point of the problem
        /// </summary>
        Point* betterReferencePoint;
        /// <summary>
        /// maximum time of estimation in ms
        /// </summary>
        int MaxEstimationTime;
        /// <summary>
        /// Custom random seed
        /// </summary>
        unsigned seed = 0;
        /// <summary>
        /// default constructor
        /// </summary>
        /// <param name="set"></param>
        /// <param name="referencePointsCalculationStyle"></param>
        /// <param name="TurnOnMonteCarloo"></param>
        /// <param name="MaxEstimationTime"></param>
        /// <param name="mcSettings"></param>
        /// <param name="callbacks"></param>
        /// <param name="SearchSubject"></param>
        /// <param name="sort"></param>
        /// <param name="del"></param>
        /// <param name="iterationsLimit"></param>
        

        SolverParameters(ReferencePointCalculationStyle worseReferencePointCalculationStyle = ReferencePointCalculationStyle::epsilon, ReferencePointCalculationStyle betterReferencePointCalculationStyle = ReferencePointCalculationStyle::epsilon,
            int MaxEstimationTime = 1000,  bool callbacks = true);
        SolverParameters();

    };

    class QEHCParameters : public SolverParameters {
    public : 
        enum SearchSubjectOption {
            MinimumContribution,
            MaximumContribution,
            Both
        };
        /// <summary>
        /// iterations limit for QEHCSolver
        /// </summary>
        unsigned long int iterationsLimit = 1000;
        /// <summary>
        /// If sorting is not allowed in QEHCSolver, should the set be shuffled or rotated by an offset
        /// </summary>
        bool shuffle = true;
        /// <summary>
        /// If set is being rotated in QEHCSolver, indicates rotation offset
        /// </summary>
        int offset = 2;
        /// <summary>
        /// type of the problem for the QEHCSolver contribution (if true, QEHCSolver will return minimum contribution; if false contrary)
        /// </summary>
        SearchSubjectOption SearchSubject = SearchSubjectOption::MinimumContribution;
        /// <summary>
        /// Is sorting allowed in QEHCSolver
        /// </summary>
        bool sort = true;
        int maxlevel = 10;
        



        QEHCParameters(ReferencePointCalculationStyle worseReferencePointCalculationStyle = ReferencePointCalculationStyle::epsilon, ReferencePointCalculationStyle betterReferencePointCalculationStyle = ReferencePointCalculationStyle::epsilon,
            int MaxEstimationTime = 1000,  bool callbacks = true, unsigned long int iterationsLimit = 1000, bool sort = true, 
            QEHCParameters::SearchSubjectOption minimalContribution = QEHCParameters::SearchSubjectOption::MinimumContribution, bool shuffle = true, int offset = 2);
    };

    class HSSParameters : public SolverParameters {
    public:
        enum StoppingCriteriaType {
            SubsetSize,
            Time
        };
        enum SubsetSelectionStrategy {
            Incremental,
            Decremental
        };

        /// <summary>
        /// Time after which HSSSolver stops.
        /// </summary>
        int StoppingTime;
        /// <summary>
        /// Number of points in a subset after which the HSSSolver stops.
        /// </summary>
        int StoppingSubsetSize;
        /// <summary>
        /// Type of criteria for Subset Selection stopping
        /// 1 - subset size
        /// 2 - selection time
        /// to be reimplemented as enum
        /// </summary>
        StoppingCriteriaType StoppingCriteria;
        /// <summary>
        /// Is it incremental version of the algorithm (decremental if false)
        /// </summary>
        SubsetSelectionStrategy Strategy;
        /// <summary>
        /// Is this method Experimental (this is just a boolean marker, to distinguish method names)
        /// </summary>
        bool Experimental = false;
        /// <summary>
        /// In the process of subset selection, should the HV be calculated.
        /// </summary>
        bool CalculateHV = false;


        HSSParameters(ReferencePointCalculationStyle worseReferencePointCalculationStyle = ReferencePointCalculationStyle::epsilon, ReferencePointCalculationStyle betterReferencePointCalculationStyle = ReferencePointCalculationStyle::epsilon,
            int MaxEstimationTime = 1000,  bool callbacks = true, HSSParameters::SubsetSelectionStrategy SubsetSelectionQHVIncremental = HSSParameters::SubsetSelectionStrategy::Decremental, HSSParameters::StoppingCriteriaType StoppingCriteria = HSSParameters::StoppingCriteriaType::SubsetSize, int StopTime = 1000, int StopSize = 100);
    };

    class DBHVEParameters : public SolverParameters {
    public:


        /// <summary>
        /// number of monte carlo iterations
        /// </summary>
        unsigned MCiterations = 100;

        DBHVEParameters(ReferencePointCalculationStyle worseReferencePointCalculationStyle = ReferencePointCalculationStyle::epsilon, ReferencePointCalculationStyle betterReferencePointCalculationStyle = ReferencePointCalculationStyle::epsilon,
            int MaxEstimationTime = 1000,  bool callbacks = true);
    };

    class QHV_BQParameters : public SolverParameters {

    public:
        /// <summary>
        /// Monte Carlo parameters for MC based methods
        /// </summary>
        SwitchParameters SwitchToMCSettings;
        bool MonteCarlo = false;



        QHV_BQParameters(ReferencePointCalculationStyle worseReferencePointCalculationStyle = ReferencePointCalculationStyle::epsilon, ReferencePointCalculationStyle betterReferencePointCalculationStyle = ReferencePointCalculationStyle::epsilon,
            int MaxEstimationTime = 1000,  bool callbacks = true, SwitchParameters mcSettings = SwitchParameters::defaultSettings(), bool TurnOnMonteCarlo = false);
    };
    class QHV_BRParameters : public SolverParameters {

    public:

        QHV_BRParameters(ReferencePointCalculationStyle worseReferencePointCalculationStyle = ReferencePointCalculationStyle::epsilon, ReferencePointCalculationStyle betterReferencePointCalculationStyle = ReferencePointCalculationStyle::epsilon,
            int MaxEstimationTime = 1000,  bool callbacks = true);
    };

    class IQHVParameters : public SolverParameters {
    public:

        IQHVParameters(ReferencePointCalculationStyle worseReferencePointCalculationStyle = ReferencePointCalculationStyle::epsilon, ReferencePointCalculationStyle betterReferencePointCalculationStyle = ReferencePointCalculationStyle::epsilon,
            int MaxEstimationTime = 1000,  bool callbacks = true);
    };

    class MCHVParameters : public SolverParameters {
    public:
        MCHVParameters(ReferencePointCalculationStyle worseReferencePointCalculationStyle = ReferencePointCalculationStyle::epsilon, ReferencePointCalculationStyle betterReferencePointCalculationStyle = ReferencePointCalculationStyle::epsilon,
            int MaxEstimationTime = 1000,  bool callbacks = true);
    };

    class QR2Parameters : public SolverParameters {
    public:
        /// <summary>
        /// In the process of R2 calculation, should hypervolume be calculated 
        /// </summary>
        bool CalculateHV;
        QR2Parameters(ReferencePointCalculationStyle worseReferencePointCalculationStyle = ReferencePointCalculationStyle::epsilon, ReferencePointCalculationStyle betterReferencePointCalculationStyle = ReferencePointCalculationStyle::epsilon,
            int MaxEstimationTime = 1000, bool callbacks = true, bool calculateHV = false);
    };


}
#endif // C_SOLVER_SETTINGS


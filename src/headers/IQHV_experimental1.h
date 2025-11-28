#include "include.h"
#include "Result.h"
#include "DataSet.h"
#include "Solver.h"
#include "SolverSettings.h"
#ifndef C_IQHV_EXP1
#define C_IQHV_EXP1

namespace qhv {
    /// <summary>
    /// This version of IQHV directly adapted from 2024 repository. Experimental version, currently not being developed. Serves as a code source.
    /// </summary>
    class IQHV_experimental1 : public Solver 
    {
    public:
        /// <summary>
        /// in the future this method will be deleted. Solver Settings are created ad-hoc
        /// </summary>
        /// <param name="problem">problem to be solved</param>
        /// <returns>solution</returns>
        Result* Solve(DataSet* problem);
        /// <summary>
        /// default solve function
        /// </summary>
        /// <param name="problem">problem to be solved</param>
        /// <param name="parameters">solver parameters, such us worse and better reference points</param>
        /// <returns>solution</returns>
        Result* Solve(DataSet* problem, SolverSettings settings);

    private:

        DType initAndSolveIQHV(vector<Point*>& set, Point& idealPoint, Point& nadirPoint, int numberOfSolutions,
            clock_t runtimeLimit = INT_MAX);
        DType solveIQHV(int start, int end, Point& IdealPoint, Point& NadirPoint, short offset, clock_t runtimeLimit, int recursion, int outerIteratorValue);

    };
}
#endif // C_IQHV_EXP1


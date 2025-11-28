#include "include.h"
#include "Result.h"
#include "DataSet.h"
#include "Solver.h"
#include "SolverSettings.h"
#ifndef C_IQHV_EXP2
#define C_IQHV_EXP2

namespace qhv {
    /// <summary>
    /// This version of IQHV directly adapted from 2018 repository. Experimental version, currently not being developed. Serves as a code source.
    /// </summary>
    class IQHV_experimental2 : public Solver
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
            bool bRemoveDominated, bool cut);
        
        DType solveIQHV(int start, int end, Point& IdealPoint, Point& NadirPoint, int offset, bool bRemoveDominated, bool cut, int recursion, int outerIteratorValue);

        Point* point2;
    };
}
#endif // C_IQHV_EXP2


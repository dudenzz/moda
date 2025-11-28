#include "include.h"
#include "Result.h"
#include "DataSet.h"
#include "Solver.h"
#include "SolverSettings.h"
#ifndef C_IQHV_CONTR
#define C_IQHV_CONTR

namespace qhv {
    /// <summary>
    /// This is the current solution for IQHV, which is actively being worked on.
    /// </summary>
    class IQHV_Contribution : public Solver
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



        DType solveIQHV_contribution(int newPoint, vector <Point*>& set, Point& IdealPoint, Point& NadirPoint, int numberOfSolutions, clock_t runtimeLimit = INT_MAX, bool callbacks = false);
        DType IQHV_callbacks(int start, int end, Point& IdealPoint, Point& NadirPoint, short offset, int recursion, int outerIteratorValue);
        DType IQHV(int start, int end, Point& IdealPoint, Point& NadirPoint, short offset);

    };
}
#endif // C_IQHV_Approx


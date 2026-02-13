#pragma once
#include "include.h"
#include "Result.h"
#include "DataSet.h"
#include "Solver.h"
#include "SolverParameters.h"
#include "DBHVE.h"
#ifndef C_HVE
#define C_HVE

namespace moda {
    /// <summary>
    /// This is the current solution for IQHV_for_hss, which is actively being worked on.
    /// </summary>
    class DBHVESolver : public Solver
    {
    public:
        /// <summary>
        /// default solve function
        /// </summary>
        /// <param name="problem">problem to be solved</param>
        /// <param name="parameters">solver parameters, such us worse and better reference points</param>
        /// <returns>solution</returns>
        DBHVEResult* Solve(DataSet* problem, DBHVEParameters settings);

    private:

        DType initAndSolveHVE(std::vector <Point*>& set, Point& idealPoint, Point& nadirPoint, int numberOfSolutions, int iterations, clock_t runtimeLimit,
            unsigned seed, bool callbacks);
    };


}
#endif // C_IQHV_Approx


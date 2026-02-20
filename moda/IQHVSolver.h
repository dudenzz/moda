#pragma once
#include "include.h"
#include "Result.h"
#include "DataSet.h"
#include "Solver.h"
#include "SolverParameters.h"

namespace moda {
    /// <summary>
    /// This is the current solution for IQHV_for_hss, which is actively being worked on.
    /// </summary>
    class IQHVSolver : public Solver
    {
    public:
        /// <summary>
        /// default solve function
        /// </summary>
        /// <param name="problem">problem to be solved</param>
        /// <param name="parameters">solver parameters, such us worse and better reference points</param>
        /// <returns>solution</returns>
        HypervolumeResult* Solve(DataSet* problem, IQHVParameters settings);

    private:

        DType initAndSolveIQHV(Point& idealPoint, Point& nadirPoint, int numberOfSolutions,  bool callbacks = false);

        
        
    };
}


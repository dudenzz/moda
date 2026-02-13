#pragma once
#include "include.h"
#include "Result.h"
#include "DataSet.h"
#include "Solver.h"
#include "SolverParameters.h"
#include "queue"


namespace moda {
    /// <summary>
    /// Hypervolume subset selection, methods from Greedy Decremental Quick Hypervolume Subset
    /// Selection Algorithms 2022, A. Jaszkiewicz, P. Zielniewicz; Parallel Problem Solving from Nature – PPSN XVII
    /// </summary>
    class HSSSolver : public Solver
    {
    public:

        /// <summary>
        /// default solve function
        /// </summary>
        /// <param name="problem">problem to be solved</param>
        /// <param name="parameters">solver parameters, such us worse and better reference points</param>
        /// <returns>solution</returns>
        HSSResult* Solve(DataSet* problem, HSSParameters settings);
    };


}


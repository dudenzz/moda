#pragma once
#include "include.h"
#include "Result.h"
#include "DataSet.h"
#include "Solver.h"
#include "SolverParameters.h"
#include "SubProblemsStackLevel.h"
#include "SPData.h"
#include "NDTree.h"
#include "TreeNode.h"


namespace moda {
    /// <summary>
    /// This version of Monte Carlo method from  directly adapted from 2018 repository. Experimental version, currently not being developed. Serves as a code source.
    /// </summary>
    class MCHVESolver : public Solver
    {
    public:

        /// <summary>
        /// default solve function
        /// </summary>
        /// <param name="problem">problem to be solved</param>
        /// <param name="parameters">solver parameters, such us worse and better reference points</param>
        /// <returns>solution</returns>
        MCHVResult* Solve(DataSet* problem, MCHVParameters settings);
    };
}


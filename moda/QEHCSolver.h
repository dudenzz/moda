#include "include.h"
#include "Result.h"
#include "DataSet.h"
#include "Solver.h"
#include "SolverParameters.h"
#include "myvector.h"
#include "ProcessData.h"
#include "SubproblemsPool.h"
#include "SubproblemParam.h"
#ifndef C_QEHC
#define C_QEHC

namespace moda {
    /// <summary>
    /// This is the current solution for IQHV_for_hss, which is actively being worked on.
    /// </summary>
    class QEHCSolver : public Solver
    {
    public:

        /// <summary>
        /// default solve function
        /// </summary>
        /// <param name="problem">problem to be solved</param>
        /// <param name="parameters">solver parameters, such us worse and better reference points</param>
        /// <returns>solution</returns>
        QEHCResult* Solve(DataSet* problem, QEHCParameters settings);

    private:

        
        QEHCResult solveQEHC(std::vector <Point*>& set, int numberOfSolutions, QEHCParameters::SearchSubjectOption seachSubject,  bool useSort = true, bool useShuffle = true, int offset = 2, int maxlevel=10, unsigned long int iterationLimit = 1);
        myvector<Point*> indexSet;
    };
}
#endif // C_QEHC


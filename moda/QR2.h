#pragma once
#pragma once
#include "include.h"
#include "Result.h"
#include "DataSet.h"
#include "Solver.h"
#include "SolverParameters.h"
#include "queue"


namespace moda {

    class Range {
    public:
        DType yMin;
        DType yMax;
        short setJSize;
        short setJ[16];
    };
    /// <summary>
    /// Quick R2 calculation, methods from Exact Calculation and Properties of the R2 Multiobjective Quality Indicator
    /// Transactions on Evolutionary Computation 2024, A. Jaszkiewicz, P. Zielniewicz;
    /// </summary>
    class QR2Solver : public Solver
    {
    public:

        /// <summary>
        /// default solve function
        /// </summary>
        /// <param name="problem">problem to be solved</param>
        /// <param name="parameters">solver parameters, such us worse and better reference points</param>
        /// <returns>solution</returns>
        R2Result* Solve(DataSet* problem, QR2Parameters settings);

    private:

        //      DType initAndSolveIQHV(vector<Point*>& set, Point& idealPoint, Point& nadirPoint, int numberOfSolutions,
        //          clock_t runtimeLimit = INT_MAX, bool callbacks = false);
        //      DType solveIQHV(int start, int end, Point& IdealPoint, Point& NadirPoint, short offset);
        //      DType solveIQHVWithCallbacks(int start, int end, Point& IdealPoint, Point& NadirPoint, short offset, int recursion, int outerIteratorValue);


        DType calculateR2(short h);
        DType calculateR2(short h, Range& range);
        DType calculateR2Tentative(const Point& nadirPoint, const Point& point, const Point& idealPoint, bool calculateHV);
        DType calculateR2Contribution(const Point& nadirPoint, const Point& point, const Point& idealPoint, bool calculateHV);
        DType QR2contribution(int start, int end, Point& idealPoint, Point& nadirPoint, unsigned offset, R2Result* result, bool calculateHV);
        void QR2init(int start, int end, Point& idealPoint, Point& nadirPoint, unsigned offset, R2Result* result, bool calculateHV);
        R2Result* solveQR2(std::vector <Point*>& points, Point& idealPoint, Point& nadirPoint, bool calculateHV);
        // Original recursive version of IQHV_for_hss (Improved QHV)
        DType qr2HyperVolume;


        long long NumberOfSubproblems;
        int maxIndexUsed;


        Point ReferencePoint;
        DType yMin[16];
        DType yMax[16];
        DType t[16];




        short potentialMins[16];
        Range ranges[16];
        DType yMinPow[16];
        DType yMaxPow[16];

        DType inf = std::numeric_limits<DType>::infinity();

    };


}


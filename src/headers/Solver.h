#include <limits.h>
#include "DataSet.h"
#include "DataSetParameters.h"
#include "SolverParameters.h"
#include <stdexcept>


#ifndef C_SOLVER
#define C_SOLVER


namespace moda {

    class Solver
    {
    public:
        //Base constructor
        Solver();

        
        // przygotowac funkcje preparuj¹ce dane
        // Problem specificaaation
        DataSet* currentlySolvedProblem;
        DataSetParameters* currentSettings;
        //Solve functions
        //virtual Result* Solve(DataSet* problem) = 0;
        //virtual Result* Solve(DataSet* problem, SolverParameters settings) = 0;

        //Callbacks
        void (*StartCallback)(DataSetParameters problemSettings, std::string SolverMessage);
        void (*IterationCallback)(int currentIteration, int totalIterations, Result*stepResult);
        void (*EndCallback)(DataSetParameters problemSettings, Result *stepResult);

        //Misc
        DType Hypervolume(const Point *nadirPoint, const Point *idealPoint);
        DType Hypervolume(const Point *nadirPoint, const Point *p2, const Point *idealPoint);
        DType Volume2(const Point* nadirPoint, const Point* p2, const Point* idealPoint);


        
        void RemoveDominated(int& start, int& end, Point IdealPoint);
        void Normalize(std::vector<DType>& p);
        DType Norm(std::vector<DType>& p);
    protected:
        //Deep copies of operational data
        

        Point* point2;
        Point* betterPoint;
        Point* worsePoint;


        //Misc
        long long branches;
        long long leafs;
        int maxIndexUsed;
        clock_t it0;
        Point tmpPoint;
        //empty callback
        void prepareData(DataSet* problem, SolverParameters settings);
        static void EmptyCallback(int currentIteration, int totalIterations, Result*stepResult);
        static void EmptyCallback(DataSetParameters problemSettings, Result *stepResult);
        static void EmptyCallback(DataSetParameters problemSettings, std::string SolverMessage);
        short off(short offset, short j);
        // Original recursive version of IQHV_for_hss (Improved QHV)
        DType IQHV(int start, int end, Point& idealPoint, Point& nadirPoint, unsigned offset);


        
    };
}
#endif
#pragma once

#include "Point.h"
#include "Result.h"
#include "myvector.h"
#include "Helpers.h"
#include "Solver.h"
#include "SubProblemsStackLevel.h"
#include "SPData.h"
#include "NDTree.h"
namespace moda {
	/// <summary>
/// This class represents code for the paper "Approximate Hypervolume calculation with guaranteed or confidence bounds" (2020)
/// This class is dedicated for thesolver with upper and lower bounds estimation, which uses the SubProblem set.
/// Method in the paper is annotated as QHV-BQ
/// </summary>

	class QHV_BQSolver : public Solver {
	public :
		
		QHV_BQResult* Solve(DataSet* problem, QHV_BQParameters settings);
	private :
		long long branches;
		Point* point2;
		myvector<Point*> indexSet;
		int maxIndexUsed = 0;
		int callbackViewer;
		int callbackViewerMax = 1000000;
		
		DType lowerBoundVolume;
		DType upperBoundVolume;
		inline int off(int offset, int j);
		// Entry point to QHV with bounds calculation using priority queue of subproblems
		std::tuple<DType, DType, DType>  solveQHV_BQ(std::vector <Point*>& set, Point& idealPoint, Point& nadirPoint, int numberOfSolutions, clock_t maxTime, bool switch2MC,
			std::vector <QHV_BQResult*>& results, SwitchParameters mcSettings);
		// standard Monte Carlo sampling working on subProblemsStack
		
			
		SubproblemsPool<SubProblem>* subProblems;
	};
}
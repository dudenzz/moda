#pragma once
#include "Point.h"
#include "Helpers.h"
#include "Hypervolume.h"
#include "SolverParameters.h"
#include <future>
#include "Helpers.h"
#include "Hypervolume.h"
#include "SubproblemsPool.h"
#include "MemoryManager.h"
#include "SubProblem.h"
#include "SubproblemParam.h"
#include "ProcessData.h"

namespace moda {
	namespace backend {
		std::tuple<DType, DType, DType>  QHV_BQ(std::vector <Point*>& set, Point& idealPoint, Point& nadirPoint, int numberOfSolutions, clock_t maxTime, bool switch2MC,
			std::vector < QHV_BQResult* >& results, SwitchParameters mcSettings, int numberOfObjectives);
		// standard Monte Carlo sampling working on subProblemsStack
		std::tuple<DType, DType, DType>  solveMC(std::vector <Point*>& allSolutions, SubProblemsStackLevel& subProblemsStack, clock_t maxTime, clock_t t0, clock_t time,
			std::vector <QHV_BQResult*>& results, SubproblemsPool<SubProblem>* subProblems, DType upperBoundVolume, DType lowerBoundVolume, int numberOfObjectives);
	}
}
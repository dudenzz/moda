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
#include <climits>
// this is an alternative implementation of the same algorithm for QEHC 
// this implementation is not parallelized yet
namespace moda {
	namespace backend {
		void innerFunc(int contextId, int iterLimit,  int offset, bool useShuffle, bool useSort, int numberOfObjectives, QEHCParameters::SearchSubjectOption searchSubject, int maxIndexUsed, DType& maxContributionLowerBound, DType& minContributionUpperBound, int& lowerBoundProcessId, int& upperBoundProcessId);

		QEHCResult QEHC_parallel(int memorySlot, int numberOfSolutions, int maxlevel, QEHCParameters::SearchSubjectOption searchSubject, bool useSort, bool useShuffle, int offset, unsigned long int iterationLimit, int numberOfObjectives);
		bool sortByPointsCounterAsc(SubproblemParam lhs, SubproblemParam rhs);
	}
}
#pragma once
#include "Point.h"
#include "Helpers.h"
#include "Hypervolume.h"
#include "SolverParameters.h"
namespace moda {
	namespace backend {
#if CALLBACKS == 1

		DType IQHV(int start, int end, int memorySlot, Point IdealPoint, Point NadirPoint, int recursion, int numberOfObjectives, int outerIteratorValue, int fullSize, bool topLevelExecutionvoid, clock_t it0, void (*IterationCallback)(int, int, Result*));
#else
		DType IQHV(int start, int end, int memorySlot, Point IdealPoint, Point NadirPoint, int recursion, int numberOfObjectives, int outerIteratorValue, int fullSize, bool topLevelExecution);
#endif
	}
}
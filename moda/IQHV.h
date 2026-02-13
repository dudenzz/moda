#pragma once
#include "Point.h"
#include "Helpers.h"
#include "Hypervolume.h"
#include "SolverParameters.h"
namespace moda {
	namespace backend {
		DType IQHV(int start, int end, int memorySlot, Point IdealPoint, Point NadirPoint, int recursion, int numberOfObjectives, int outerIteratorValue, int fullSize);

	}
}
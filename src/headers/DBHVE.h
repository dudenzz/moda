#pragma once
#include "Point.h"
#include "NDTree.h"
#include "Hypervolume.h"
#include <random>
namespace moda {
	namespace backend {
		DType DBHVE(int memorySlot, Point& idealPoint, Point& nadirPoint, int numberOfSolutions, int MCiterations, clock_t runtimeLimit,
			unsigned seed, int numberOfObjectives, Point& referencePoint, clock_t& it0);
	}
}
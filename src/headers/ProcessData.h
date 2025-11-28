#pragma once
#include "SubProblemsStackLevel.h"

namespace moda {
	class ProcessData {
	public:

		SubProblemsStackLevel subProblemsStack;
		
		ProcessData(int maxlevel) {
			subProblemsStack = SubProblemsStackLevel(maxlevel);
		}
		DType upperBoundVolume;
		DType lowerBoundVolume;
		DType totalVolume;
		int id;
	};
}

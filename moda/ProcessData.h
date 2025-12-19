#pragma once
#include "SubProblemsStackLevel.h"

namespace moda {
	class ProcessData {
	public:


		
		ProcessData(int maxlevel): subProblemsStack(maxlevel) { }
		DType upperBoundVolume;
		DType lowerBoundVolume;
		DType totalVolume;
		int id;
		SubProblemsStackLevel subProblemsStack;
	};
}

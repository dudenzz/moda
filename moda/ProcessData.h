#pragma once
#include "SubProblemsStackLevel.h"

namespace moda {
	class ProcessData {
	public:


		ProcessData() = default;
		ProcessData(const ProcessData& other) = default;
		ProcessData(ProcessData* other) {

		};
		ProcessData(int maxlevel): subProblemsStack(maxlevel) { }
		DType upperBoundVolume;
		DType lowerBoundVolume;
		DType totalVolume;
		int id;
		SubProblemsStackLevel subProblemsStack;
	};
}

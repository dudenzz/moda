#pragma once
#include "SubProblem.h"
#include "SubproblemsPool.h"
namespace moda {
	struct volumeCompare {
		bool operator() (int i1, int i2) const {
			return subProblems[i1].level > subProblems[i2].level;
		}
	};
	class SubProblemsStackPriorityQueue : public priority_queue <int, vector<int>, volumeCompare> {
	public:
		
		void push_back(int subProblem) {
			subProblems[subProblem].level = logarithm(subProblems[subProblem].volume, div_qehc);
			push(subProblem);
		}

		int back() {
			return top();
		}

		void pop_back() {
			pop();
		}
	};
}
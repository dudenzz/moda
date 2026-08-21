#include "SubProblemsStackPriorityQueue.h"

namespace moda {
	void SubProblemsStackPriorityQueue::push_back(int subProblem)
    {
        subProblems[subProblem].level = logarithm(subProblems[subProblem].volume, div_qehc);
        push(subProblem);
    }

}
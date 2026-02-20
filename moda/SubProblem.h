#pragma once
#include "Point.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <set>
#include <ctime>
#include <string>
#include <sstream>
#include <algorithm>
#include <queue>
#include <random>
#include <iomanip> 

namespace moda {

	class SubProblem {
	public:
		DType volume;
		int level;
		Point IdealPoint;
		Point NadirPoint;
		int start;
		int end;
		SubProblem()
		{

		}
		SubProblem(const SubProblem& subproblem)
		{
			volume = subproblem.volume;
			level = subproblem.level;
			IdealPoint = subproblem.IdealPoint;
			NadirPoint = subproblem.NadirPoint;
			start = subproblem.start;
			end = subproblem.end;
		}
		SubProblem& operator=(SubProblem& subproblem)
		{
			volume = subproblem.volume;
			level = subproblem.level;
			IdealPoint = subproblem.IdealPoint;
			NadirPoint = subproblem.NadirPoint;
			start = subproblem.start;
			end = subproblem.end;
			return *this;
		}
	};
}
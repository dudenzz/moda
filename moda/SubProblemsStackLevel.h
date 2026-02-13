#pragma once

#include "SubProblem.h"
#include "SubproblemsPool.h"


namespace moda {
	//SubproblemsPool <SubProblem> subProblems;

	class SubProblemsStackLevel {

	private:

		int _size = 0;
		int missed = 0;

		int iteratorLevel;
		int iteratorPos;
		int maxlevel = 10;
		std::vector <std::vector<int>> levels;

	public:
		std::set <int> usedLevels;
		SubproblemsPool<SubProblem>* subProblems;
		long currentLevel = 0;
		SubProblemsStackLevel() {
			levels.resize(maxlevel);
		}
		SubProblemsStackLevel(int maxlevel);


		void startIterating();
		int next();
		void push_back(int iSP);
		int back();
		void pop_back();
		int size();

		SubProblemsStackLevel(const SubProblemsStackLevel&) = delete;
		SubProblemsStackLevel& operator=(const SubProblemsStackLevel&) = delete;
	};
}
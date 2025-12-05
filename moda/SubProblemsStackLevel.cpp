#include "SubProblemsStackLevel.h"

namespace moda {

	SubProblemsStackLevel::SubProblemsStackLevel(int maxlevel) {
		this->maxlevel = maxlevel;
		levels.resize(maxlevel);
	}

	SubProblemsStackLevel::~SubProblemsStackLevel() {
		levels[currentLevel].clear();
		for (auto l : usedLevels) {
			levels[l].clear();
		}
	}

	void SubProblemsStackLevel::startIterating() {
		iteratorLevel = currentLevel;

		iteratorPos = 0;
	}

	int SubProblemsStackLevel::next()
	{
		if (levels[iteratorLevel].size() <= iteratorPos) {
			levels[iteratorLevel].clear();

			if (usedLevels.size() == 0) {
				return -1;
			}

			std::set<int>::iterator it = usedLevels.begin();
			iteratorLevel = *(it);
			usedLevels.erase(it);

			iteratorPos = 0;
		}
		return levels[iteratorLevel][iteratorPos++];
	}

	void SubProblemsStackLevel::push_back(int iSP) {
		_size++;
		int level = logarithm((*subProblems)[iSP].volume, div_qehc);
		if (level > maxlevel - 1) {
			missed++;
			level = maxlevel - 1;
		}
		if (level > currentLevel) {
			if (levels[level].size() == 0) {
				usedLevels.insert(level);
			}
			levels[level].push_back(iSP);
		}
		else {
			levels[currentLevel].push_back(iSP);
		}
	}
	int SubProblemsStackLevel::back() {
		if (levels[currentLevel].size() == 0) {
			std::set<int>::iterator it = usedLevels.begin();
			currentLevel = *(it);

			usedLevels.erase(it);
#if VERBOSE >= 1
			if (currentLevel == MAXLEVEL - 1) {

				cout << "* ";

			}
#endif
		}
		return levels[currentLevel].back();

	}

	void SubProblemsStackLevel::pop_back() {
		levels[currentLevel].pop_back();
		_size--;
	}

	int SubProblemsStackLevel::size() {
		return _size;
	}


}
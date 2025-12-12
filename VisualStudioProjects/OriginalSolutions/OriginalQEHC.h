#pragma once
#include "pch.h"
#include <vector>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <queue>
#include <iostream>
#include <algorithm>
#include <set>
#include <random>

using namespace std;


namespace OriginalQEHC {
#define MAXLEVEL 3
#define div  0.995
	double logarithm(double param, double base) {
		return log(param) / log(base);
	}
	int NumberOfObjectives;

	/** Possible relations between points in multiple objective space */
	enum TCompare { _Dominating, _Dominated, _Nondominated, _EqualSol };

	/** Point in objective space */
	class TPoint {
	public:
		/** Vector of objective values */
		float ObjectiveValues[10];


		/** Copy operator */
		TPoint& operator = (TPoint& Point) {
			int i; for (i = 0; i < NumberOfObjectives; i++)
				ObjectiveValues[i] = Point.ObjectiveValues[i];
			return *this;
		}

		/** Compares two points in multiple objective space
		*
		*	Returns _Dominated if this is dominated
		*	Returns _Dominating if this is dominating
		*	It may also return _Nondominated or _EqualSol */
		TCompare Compare(TPoint& Point) {
			bool bBetter = false;
			bool bWorse = false;

			int i = 0;
			do {
				if (ObjectiveValues[i] > Point.ObjectiveValues[i])
					bBetter = true;
				if (Point.ObjectiveValues[i] > ObjectiveValues[i])
					bWorse = true;
				i++;
			} while (!(bWorse && bBetter) && (i < NumberOfObjectives));
			if (bWorse) {
				if (bBetter) {
					return _Nondominated;
				}
				else {
					return _Dominated;
				}
			}
			else {
				if (bBetter) {
					return _Dominating;
				}
				else {
					return _EqualSol;
				}
			}
		}



		/** Reads the point from the stream */
		istream& Load(istream& Stream) {
			int i; for (i = 0; i < NumberOfObjectives; i++) {
				Stream >> ObjectiveValues[i];

				ObjectiveValues[i] = ObjectiveValues[i];
			}
			return Stream;
		}

		/** Saves objective values to an open Stream
		*
		*	Values are separated by TAB character */
		ostream& Save(ostream& Stream) {
			int i;
			for (i = 0; i < NumberOfObjectives; i++) {
				Stream << ObjectiveValues[i];
				Stream << '\x09';
			}
			Stream << '\x09';
			return Stream;
		}

		double Distance(TPoint& Point, TPoint& IdealPoint, TPoint& NadirPoint) {
			double s = 0;
			int iobj; for (iobj = 0; iobj < NumberOfObjectives; iobj++) {
				double Range = IdealPoint.ObjectiveValues[iobj] - NadirPoint.ObjectiveValues[iobj];
				if (Range == 0)
					Range = 1;
				double s1 = (ObjectiveValues[iobj] - Point.ObjectiveValues[iobj]) / Range;
				s += s1 * s1;
			}
			return sqrt(s);
		}

	};
	template <class T>
	class myvector {
	protected:
		int shift = 22;
		int base = (1 << shift);
		int mask = base - 1;
		vector <vector <T>> vec;

		int row;
		int col;

		int currentMaxRow = 0;

	public:
		myvector() {
			vec.resize(base);
		}

		~myvector() {
			vec.clear();
		}

		void reserve(int newsize) {
		}

		int size() {
			return (currentMaxRow + 1) * base;
		}

		void resize(int newSize) {
			if (newSize < size()) {
				currentMaxRow = 0;
			}

			int newRows = 1 + (newSize - size()) / base;
			int i;
			for (i = currentMaxRow; i <= currentMaxRow + newRows; i++) {
				vec[i].resize(base);
			}
			currentMaxRow += newRows;
		}

		inline T& operator[](const int index) {
			return vec[index >> shift][index & mask];
		}

		inline const T& get(const int index) {
			return vec[index >> shift][index & mask];
		}

		void clear() {
			currentMaxRow = 0;
		}
	};

	void Load(vector <TPoint*>& set, istream& Stream)
	{
		int i = 0;

		char nextChar = Stream.get();
		Stream.unget();

		if (nextChar == '#') {
			// Read the rest of the line
			char c;
			do {
				Stream.get(c);
			} while (c != '\n');
		}
		nextChar = '0';

		while (nextChar >= '0' && nextChar <= '9' && set.size() < 100000) {
			i++;
			TPoint* Solution = new TPoint;

			Solution->Load(Stream);

			if (Stream.rdstate() == ios::goodbit) {
				set.push_back(Solution);
				// Read the rest of the line
				char c;
				do {
					Stream.get(c);
				} while (c != '\n');
			}
			else
				delete Solution;

			nextChar = Stream.get();
			Stream.unget();
		}
	}

	myvector <TPoint*> indexSet;
	vector <TPoint*> indexSetVec;
	TPoint* point2;
	int maxIndexUsed = 0;
	struct SubProblem {
		double volume;
		int level;
		TPoint IdealPoint;
		TPoint NadirPoint;
		int start;
		int end;
	};

	// Class used for subproblems memory management intead of standard new and delete
	template <class T>
	class SubproblemsPool : public myvector <T> {
	private:
		int maxElementUsed = -1;

		priority_queue<int> pool;

	public:
		SubproblemsPool() {
			clock_t t0 = clock();
			this->reserve(100000000);
			this->resize(1000000);
		}
		int getNew() {
			if (pool.size() > 0) {
				int index = pool.top();
				pool.pop();
				return index;
			}
			else {
				if (maxElementUsed >= this->size() - 1) {
					this->resize(this->size() + 1000000); // !!! 10000000
				}
				return ++maxElementUsed;
			}
		}

		void free(int index) {
			pool.push(index);
		}

		void clear() {
			maxElementUsed = -1;
			pool = {};
			myvector<T>::clear();
		}
	};


	SubproblemsPool <SubProblem> subProblems;
	class SubProblemsStackLevel {
	private:
		int _size = 0;
		int missed = 0;

		int iteratorLevel;
		int iteratorPos;

		vector <vector<int>> levels;

	public:
		set <int> usedLevels;

		int currentLevel = 0;

		SubProblemsStackLevel() {
			levels.resize(MAXLEVEL);
		}

		~SubProblemsStackLevel() {
			levels[currentLevel].clear();
			for (auto l : usedLevels) {
				levels[l].clear();
			}
		}

		void startIterating() {
			iteratorLevel = currentLevel;

			iteratorPos = 0;
		}

		int next() {
			if (levels[iteratorLevel].size() <= iteratorPos) {
				levels[iteratorLevel].clear();

				if (usedLevels.size() == 0) {
					return -1;
				}

				set<int>::iterator it = usedLevels.begin();
				iteratorLevel = *(it);
				usedLevels.erase(it);

				iteratorPos = 0;
			}
			return levels[iteratorLevel][iteratorPos++];
		}

		void push_back(int iSP) {
			_size++;
			int level = logarithm(subProblems[iSP].volume, div);
			if (level > MAXLEVEL - 1) {
				missed++;
				level = MAXLEVEL - 1;
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

		int back() {
			if (levels[currentLevel].size() == 0) {
				set<int>::iterator it = usedLevels.begin();
				currentLevel = *(it);
				usedLevels.erase(it);
			}
			return levels[currentLevel].back();
		}

		void pop_back() {
			levels[currentLevel].pop_back();
			_size--;
		}

		int size() {
			return _size;
		}
	};
	struct SubproblemParam {
		int objectiveIndex;
		int pointsCounter;
		int partStart;
		int partEnd;
	};
	class ProcessData {
	public:
		SubProblemsStackLevel subProblemsStack;
		double upperBoundVolume;
		double lowerBoundVolume;
		double totalVolume;
		int id;
	};

	double volume(const TPoint& nadirPoint, const TPoint& idealPoint) {
		double s = 1;
		int j;
		for (j = 0; j < NumberOfObjectives; j++) {
			s *= (idealPoint.ObjectiveValues[j] - nadirPoint.ObjectiveValues[j]);
		}
		return s;
	}

	double volume(const TPoint& nadirPoint, const TPoint& p2, const TPoint& idealPoint) {
		double s = 1;
		int j;
		for (j = 0; j < NumberOfObjectives; j++) {
			s *= (min(p2.ObjectiveValues[j], idealPoint.ObjectiveValues[j]) - nadirPoint.ObjectiveValues[j]);
		}
		return s;
	}


	inline int off(int offset, int j) {
		return (j + offset) % NumberOfObjectives;
	}
	bool sortByPointsCounterAsc(SubproblemParam lhs, SubproblemParam rhs) {
		return lhs.pointsCounter < rhs.pointsCounter;
	}
	double solveQEHC(vector <TPoint*>& set, int numberOfSolutions, bool minimize, clock_t& elapsedTime1,
		clock_t& elapsedTime2, int& processId, bool useDelete = true, bool useSort = true, unsigned long int iterationLimit = 1) {

		if (!useDelete) {
			iterationLimit = ULONG_MAX;
		}

		clock_t t0 = clock();

		indexSet.resize(40000000);
		point2 = new TPoint;

		TPoint newNadirPoint;
		int j;
		for (j = 0; j < NumberOfObjectives; j++) {
			newNadirPoint.ObjectiveValues[j] = 0;
		}

		double maxContributionLowerBound = 0;
		double minContributionUpperBound = 1;
		int lowerBoundProcessId = -1;
		int upperBoundProcessId = -1;

		int pos = 0;
		unsigned long int iterLimit;
		vector <ProcessData*> processes;
		processes.resize(numberOfSolutions);

		int ip;
		int ip0;
		unsigned long int ii;
		for (ip = 0; ip < numberOfSolutions; ip++) {
			processes[ip] = new ProcessData;
			int iSP = subProblems.getNew();
			subProblems[iSP].IdealPoint = *(set[ip]);
			subProblems[iSP].NadirPoint = newNadirPoint;
			subProblems[iSP].start = pos;
			subProblems[iSP].end = pos + numberOfSolutions - 2;
			subProblems[iSP].volume = volume(subProblems[iSP].NadirPoint, subProblems[iSP].IdealPoint);
			processes[ip]->lowerBoundVolume = 0;
			processes[ip]->upperBoundVolume = subProblems[iSP].volume;
			processes[ip]->totalVolume = subProblems[iSP].volume;
			processes[ip]->subProblemsStack.push_back(iSP);
			processes[ip]->id = ip;

			int i3 = 0;
			int i2; for (i2 = 0; i2 < numberOfSolutions; i2++) {
				if (i2 != ip) {
					if ((set)[i2] == NULL) {
						continue;
					}
					indexSet[pos + i3++] = set[i2];
				}
			}
			pos += numberOfSolutions - 1;
		}
		maxIndexUsed = pos - 1;
		int oldmaxIndexUsed = maxIndexUsed;

		if (processId > 0) {
			swap(processes[0], processes[processId]);
			iterLimit = ULONG_MAX;
		}
		else
		{
			iterLimit = iterationLimit;
		}

		// use line below together with rotate function (instead of sort function on subproblemParams)
		//int offset = 0;

		long nIterations = 0;
		bool runningPhase1 = true;
		while (processes.size() > 0) {

			for (ip = 0; ip < processes.size(); ip++) {

				if (runningPhase1) {
					if (processId < 0) {
						if (processes.size() == 1) {
							elapsedTime1 = clock() - t0;
							runningPhase1 = false;
							//cout << "  Phase #1 - process id:\t" << processes[0]->id << "\n";
							//cout << "  Phase #1 - iterations:\t" << nIterations << "\n";
						}
					}
					else if (processes[ip]->id != processId) {
						elapsedTime1 = clock() - t0;
						runningPhase1 = false;
					}
				}

				//if (!useDelete && processes.size() % 50 == 0) {
				//	cout << "#";
				//}

				for (ii = 0; ii < iterLimit; ii++) {

					if (processes[ip]->subProblemsStack.size() == 0) {
						delete processes[ip];
						processes.erase(processes.begin() + ip);
						ip--;
						break;
					}

					nIterations++;
					int iSP = processes[ip]->subProblemsStack.back();
					{
						processes[ip]->subProblemsStack.pop_back();
					}

					// use two lines below together with rotate function (instead of sort function on subproblemParams)
					//offset++;
					//offset = offset % NumberOfObjectives;

					// If there is just one point
					if (subProblems[iSP].start == subProblems[iSP].end) {
						double v = volume(subProblems[iSP].NadirPoint, *(indexSet[subProblems[iSP].start]), subProblems[iSP].IdealPoint);
						processes[ip]->lowerBoundVolume += v;
						processes[ip]->upperBoundVolume -= subProblems[iSP].volume - v;
						subProblems.free(iSP);

						if (maxContributionLowerBound < processes[ip]->totalVolume - processes[ip]->upperBoundVolume) {
							maxContributionLowerBound = processes[ip]->totalVolume - processes[ip]->upperBoundVolume;
							lowerBoundProcessId = processes[ip]->id;
						}
						if (minContributionUpperBound > processes[ip]->totalVolume - processes[ip]->lowerBoundVolume) {
							minContributionUpperBound = processes[ip]->totalVolume - processes[ip]->lowerBoundVolume;
							upperBoundProcessId = processes[ip]->id;
						}

						if (useDelete) {
							bool toDelete = false;
							if (minimize) {
								if (processes[ip]->totalVolume - processes[ip]->upperBoundVolume > minContributionUpperBound) {
									toDelete = true;
								}
							}
							else {
								if (processes[ip]->totalVolume - processes[ip]->lowerBoundVolume < maxContributionLowerBound) {
									toDelete = true;
								}
							}
							if (toDelete) {
								delete processes[ip];
								processes.erase(processes.begin() + ip);
								ip--;
								break;
							}
						}
						continue;
					}

					// If there are just two points
					if (subProblems[iSP].end - subProblems[iSP].start == 1) {
						double v = volume(subProblems[iSP].NadirPoint, *(indexSet[subProblems[iSP].start]), subProblems[iSP].IdealPoint);
						v += volume(subProblems[iSP].NadirPoint, *(indexSet[subProblems[iSP].end]), subProblems[iSP].IdealPoint);
						*point2 = *(indexSet[subProblems[iSP].start]);
						unsigned j;
						for (j = 0; j < NumberOfObjectives; j++) {
							point2->ObjectiveValues[j] = min(point2->ObjectiveValues[j], indexSet[subProblems[iSP].end]->ObjectiveValues[j]);
						}
						v -= volume(subProblems[iSP].NadirPoint, *point2, subProblems[iSP].IdealPoint);

						processes[ip]->lowerBoundVolume += v;
						processes[ip]->upperBoundVolume -= subProblems[iSP].volume - v;
						subProblems.free(iSP);

						if (maxContributionLowerBound < processes[ip]->totalVolume - processes[ip]->upperBoundVolume) {
							maxContributionLowerBound = processes[ip]->totalVolume - processes[ip]->upperBoundVolume;
							lowerBoundProcessId = processes[ip]->id;
						}
						if (minContributionUpperBound > processes[ip]->totalVolume - processes[ip]->lowerBoundVolume) {
							minContributionUpperBound = processes[ip]->totalVolume - processes[ip]->lowerBoundVolume;
							upperBoundProcessId = processes[ip]->id;
						}

						if (useDelete) {
							bool toDelete = false;
							if (minimize) {
								if (processes[ip]->totalVolume - processes[ip]->upperBoundVolume > minContributionUpperBound) {
									toDelete = true;
								}
							}
							else {
								if (processes[ip]->totalVolume - processes[ip]->lowerBoundVolume < maxContributionLowerBound) {
									toDelete = true;
								}
							}
							if (toDelete) {
								delete processes[ip];
								processes.erase(processes.begin() + ip);
								ip--;
								break;
							}
						}
						continue;
					}

					processes[ip]->upperBoundVolume -= subProblems[iSP].volume;

					int iPivot = subProblems[iSP].start;
					double maxVolume;
					maxVolume = volume(subProblems[iSP].NadirPoint, *(indexSet)[iPivot], subProblems[iSP].IdealPoint);

					// Find the pivot point
					unsigned i;
					for (i = subProblems[iSP].start + 1; i <= subProblems[iSP].end; i++) {
						double volumeCurrent;
						volumeCurrent = volume(subProblems[iSP].NadirPoint, *(indexSet)[i], subProblems[iSP].IdealPoint);
						if (maxVolume < volumeCurrent) {
							maxVolume = volumeCurrent;
							iPivot = i;
						}
					}

					double v = volume(subProblems[iSP].NadirPoint, *(indexSet)[iPivot], subProblems[iSP].IdealPoint);
					processes[ip]->lowerBoundVolume += v;

					if (minContributionUpperBound > processes[ip]->totalVolume - processes[ip]->lowerBoundVolume) {
						minContributionUpperBound = processes[ip]->totalVolume - processes[ip]->lowerBoundVolume;
						upperBoundProcessId = processes[ip]->id;
					}

					if (useDelete) {
						bool toDelete = false;
						if (!minimize) {
							if (processes[ip]->totalVolume - processes[ip]->lowerBoundVolume < maxContributionLowerBound) {
								toDelete = true;
							}
						}
						if (toDelete) {
							delete processes[ip];
							processes.erase(processes.begin() + ip);
							ip--;
							break;
						}
					}

					processes[ip]->upperBoundVolume += v;

					unsigned j;
					int jj;
					int partStart;
					int partEnd;
					int iPos = maxIndexUsed + 1;

					TPoint partNadirPoint = subProblems[iSP].NadirPoint;
					TPoint partIdealPoint = subProblems[iSP].IdealPoint;
					TPoint* pPivotPoint = (indexSet)[iPivot];

					//------------------------------------------------------------------------------------------------
					SubproblemParam* subproblemParams = new SubproblemParam[NumberOfObjectives];
					for (j = 0; j < NumberOfObjectives; j++) {
						subproblemParams[j].objectiveIndex = j;
						subproblemParams[j].pointsCounter = 0;
						subproblemParams[j].partStart = iPos;

						for (i = subProblems[iSP].start; i <= subProblems[iSP].end; i++) {
							if (i == iPivot)
								continue;

							if (min(partIdealPoint.ObjectiveValues[j], indexSet[i]->ObjectiveValues[j]) >
								min(partIdealPoint.ObjectiveValues[j], (indexSet)[iPivot]->ObjectiveValues[j])) {
								indexSet[iPos++] = indexSet[i];
								subproblemParams[j].pointsCounter++;
							}
						}

						subproblemParams[j].partEnd = iPos - 1;
						maxIndexUsed = iPos - 1;
					}
					std::random_device randomDevice;
					std::mt19937 randomNumberGenerator(randomDevice());
					//------------------------------------------------------------------------------------------------
					if (useSort) {
						sort(subproblemParams, subproblemParams + NumberOfObjectives, sortByPointsCounterAsc);
					}
					else
					{
						shuffle(subproblemParams, subproblemParams + NumberOfObjectives, randomNumberGenerator);
					}
					//rotate(subproblemParams, subproblemParams + offset, subproblemParams + NumberOfObjectives);

					for (jj = 0; jj < NumberOfObjectives; jj++) {
						j = subproblemParams[jj].objectiveIndex;
						partStart = subproblemParams[jj].partStart;
						partEnd = subproblemParams[jj].partEnd;

						if (jj > 0) {
							int j2 = subproblemParams[jj - 1].objectiveIndex;
							partIdealPoint.ObjectiveValues[j2] = min(subProblems[iSP].IdealPoint.ObjectiveValues[j2],
								indexSet[iPivot]->ObjectiveValues[j2]);
							partNadirPoint.ObjectiveValues[j2] = subProblems[iSP].NadirPoint.ObjectiveValues[j2];
						}

						if (partEnd >= partStart) {
							partNadirPoint.ObjectiveValues[j] = min(subProblems[iSP].IdealPoint.ObjectiveValues[j],
								indexSet[iPivot]->ObjectiveValues[j]);

							double v = volume(partNadirPoint, partIdealPoint);
							processes[ip]->upperBoundVolume += v;

							int newISP = subProblems.getNew();
							subProblems[newISP].start = partStart;
							subProblems[newISP].end = partEnd;
							subProblems[newISP].volume = v;
							subProblems[newISP].IdealPoint = partIdealPoint;
							subProblems[newISP].NadirPoint = partNadirPoint;
							processes[ip]->subProblemsStack.push_back(newISP);
						}
					}
					delete subproblemParams;
					subProblems.free(iSP);

					if (maxIndexUsed > indexSet.size() - 10000000) {
						indexSet.resize(indexSet.size() + 10000000);
					}

					if (maxContributionLowerBound < processes[ip]->totalVolume - processes[ip]->upperBoundVolume) {
						maxContributionLowerBound = processes[ip]->totalVolume - processes[ip]->upperBoundVolume;
						lowerBoundProcessId = processes[ip]->id;
					}
					if (useDelete) {
						bool toDelete = false;
						if (minimize) {
							if (processes[ip]->totalVolume - processes[ip]->upperBoundVolume > minContributionUpperBound) {
								toDelete = true;
							}
						}
						if (toDelete) {
							delete processes[ip];
							processes.erase(processes.begin() + ip);
							ip--;
							break;
						}
					}
				}

				if (iterLimit == ULONG_MAX) {
					maxIndexUsed = oldmaxIndexUsed;
				}
				iterLimit = iterationLimit;
			}
		}

		elapsedTime2 = clock() - t0;

		//cout << "  Iterations:\t\t" << nIterations << "\n";

		if (minimize) {
			if (processId < 0) {
				processId = upperBoundProcessId;
			}
			return minContributionUpperBound;
		}
		else {
			if (processId < 0) {
				processId = lowerBoundProcessId;
			}
			return maxContributionLowerBound;
		}
	}

}
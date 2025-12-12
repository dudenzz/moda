#include "pch.h"
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

namespace OriginalIQHV
{
	using namespace std;

	int maxBranches = 5;
	int maxListSize = 20;

	int NumberOfObjectives;


	double lowerBoundVolume;
	double upperBoundVolume;

	long long branches;
	long long leafs;




	/** Possible relations between points in multiple objective space */
	enum TCompare { _Dominating, _Dominated, _Nondominated, _EqualSol };
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


	/** Result of the Hyper Volume calculation*/
	class TResult {
	public:
		/** Elapsed time */
		long ElapsedTime;

		/** Value of the Hyper Volume */
		double Volume;

		/** Lower bound of the Hyper Volume */
		double LowerBound;

		/** Upper bound of the Hyper Volume */
		double UpperBound;
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
	inline int off(int offset, int j) {
		return (j + offset) % NumberOfObjectives;
	}
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
	void normalize(vector <TPoint*>& set, TPoint& IdealPoint, TPoint& NadirPoint) {
		//Normalize the values to 0-1
		unsigned int i; for (i = 0; i < set.size(); i++) {
			if ((set)[i] == NULL) {
				continue;
			}

			int j;
			for (j = 0; j < NumberOfObjectives; j++) {
				(set)[i]->ObjectiveValues[j] = ((double)(set)[i]->ObjectiveValues[j] - NadirPoint.ObjectiveValues[j]) /
					(IdealPoint.ObjectiveValues[j] - NadirPoint.ObjectiveValues[j]);
			}
		}
	}
	vector <TPoint*> indexSetVec;
	TPoint* point2;
	int maxIndexUsed = 0;


	// Original recursive version of QHV_II (Improved QHV)
	double QHV_II(int start, int end, TPoint& IdealPoint, TPoint& NadirPoint, int offset) {
		clock_t t0 = clock();

		if (end < start) {
			return 0;
		}

		branches++;
		offset++;
		offset = offset % NumberOfObjectives;

		int oldmaxIndexUsed = maxIndexUsed;

		// If there is just one point
		if (end == start) {
			double totalVolume = volume(NadirPoint, *(indexSetVec[start]), IdealPoint);

			return totalVolume;
		}

		// If there are just two points
		if (end - start == 1) {
			double totalVolume = volume(NadirPoint, *(indexSetVec[start]), IdealPoint);
			totalVolume += volume(NadirPoint, *(indexSetVec[end]), IdealPoint);
			*point2 = *(indexSetVec[start]);
			unsigned j;
			for (j = 0; j < NumberOfObjectives; j++) {
				point2->ObjectiveValues[j] = min(point2->ObjectiveValues[j], indexSetVec[end]->ObjectiveValues[j]);
			}
			totalVolume -= volume(NadirPoint, *point2, IdealPoint);

			return totalVolume;
		}

		int iPivot = start;
		double maxVolume;
		maxVolume = volume(NadirPoint, *(indexSetVec)[iPivot], IdealPoint);

		// Find the pivot point
		unsigned i;
		for (i = start + 1; i <= end; i++) {
			double volumeCurrent;
			volumeCurrent = volume(NadirPoint, *(indexSetVec)[i], IdealPoint);
			if (maxVolume < volumeCurrent) {
				maxVolume = volumeCurrent;
				iPivot = i;
			}
		}

		double totalVolume = volume(NadirPoint, *(indexSetVec)[iPivot], IdealPoint);

		// Build subproblems
		int iPos = maxIndexUsed + 1;
		unsigned j;
		int jj;

		int ic = 0;

		TPoint partNadirPoint = NadirPoint;
		TPoint partIdealPoint = IdealPoint;

		for (jj = 0; jj < NumberOfObjectives; jj++) {
			j = off(offset, jj);

			if (jj > 0) {
				int j2 = off(offset, jj - 1);
				partIdealPoint.ObjectiveValues[j2] = min(IdealPoint.ObjectiveValues[j2], indexSetVec[iPivot]->ObjectiveValues[j2]);
				partNadirPoint.ObjectiveValues[j2] = NadirPoint.ObjectiveValues[j2];
			}

			int partStart = iPos;

			for (i = start; i <= end; i++) {
				if (i == iPivot)
					continue;

				if (min(IdealPoint.ObjectiveValues[j], indexSetVec[i]->ObjectiveValues[j]) >
					min(IdealPoint.ObjectiveValues[j], (indexSetVec)[iPivot]->ObjectiveValues[j])) {
					indexSetVec[iPos++] = indexSetVec[i];
				}

			}
			int partEnd = iPos - 1;

			maxIndexUsed = iPos - 1;

			if (partEnd >= partStart) {
				partNadirPoint.ObjectiveValues[j] = min(IdealPoint.ObjectiveValues[j], indexSetVec[iPivot]->ObjectiveValues[j]);

				totalVolume += QHV_II(partStart, partEnd, partIdealPoint, partNadirPoint, offset);
			}
		}

		maxIndexUsed = oldmaxIndexUsed;
		return (totalVolume);
	}

	// Entry point to original recursive version of QHV_II (Improved QHV)
	double solveQHV_II(vector <TPoint*>& set, TPoint& IdealPoint, TPoint& NadirPoint, int numberOfSolutions) {
		indexSetVec.resize(20000000);
		point2 = new TPoint;
		unsigned int i; for (i = 0; i < numberOfSolutions; i++) {
			if ((set)[i] == NULL) {
				continue;
			}
			indexSetVec[i] = set[i];
		}

		TPoint newIdealPoint, newNadirPoint;
		int j;
		for (j = 0; j < NumberOfObjectives; j++) {
			newIdealPoint.ObjectiveValues[j] = 1;
			newNadirPoint.ObjectiveValues[j] = 0;
		}

		maxIndexUsed = numberOfSolutions - 1;
		return QHV_II(0, numberOfSolutions - 1, IdealPoint, NadirPoint, 0);
	}
}
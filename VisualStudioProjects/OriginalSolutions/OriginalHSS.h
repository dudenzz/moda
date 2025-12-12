#pragma once
#include "pch.h"
#include <vector>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <queue>
#include <iostream>


using namespace std;

namespace OriginalHSS
{
	int NumberOfObjectives = 0;
	long long branches;
	long long leafs;
	int maxIndexUsed = 0;

	class BHC {
	public:
		unsigned contributor;
		double contribution;
	};


	struct compareUBHCMax {
		bool operator() (const BHC& uBHC1, const BHC& uBHC2) const {
			return uBHC1.contribution < uBHC2.contribution;
		}
	};


	struct compareUBHCMin {
		bool operator() (const BHC& uBHC1, const BHC& uBHC2) const {
			return uBHC1.contribution > uBHC2.contribution;
		}
	};



#pragma region point
	/** Point in objective space */
	class TPoint {
	public:
		int id = -1;

		/** Vector of objective values */
		float ObjectiveValues[10];

		/** Copy operator */
		TPoint& operator = (TPoint& Point) {
			int i; for (i = 0; i < OriginalHSS::NumberOfObjectives; i++)
				ObjectiveValues[i] = Point.ObjectiveValues[i];
			return *this;
		}

		/** Reads the point from the stream */
		std::istream& Load(std::istream& Stream) {
			for (short i = 0; i < OriginalHSS::NumberOfObjectives; i++) {
				Stream >> ObjectiveValues[i];

				ObjectiveValues[i] = ObjectiveValues[i];
			}
			return Stream;
		}

		/** Saves objective values to an open Stream
		*
		*	Values are separated by TAB character */
		std::ostream& Save(std::ostream& Stream) {
			for (short i = 0; i < OriginalHSS::NumberOfObjectives; i++) {
				Stream << ObjectiveValues[i];
				Stream << '\x09';
			}
			Stream << '\x09';
			return Stream;
		}

		short NumberOfObjectives() {
			return OriginalHSS::NumberOfObjectives;
		}
	};


#pragma endregion



#pragma region QHV_II


	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// QHV_II
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	vector <TPoint*> indexSetVec;
	TPoint tmpPoint;


	double volume(const TPoint& nadirPoint, const TPoint& idealPoint) {
		double s = 1;
		for (short j = 0; j < OriginalHSS::NumberOfObjectives; j++) {
			s *= (idealPoint.ObjectiveValues[j] - nadirPoint.ObjectiveValues[j]);
		}

		return s;
	}


	double volume(const TPoint& nadirPoint, const TPoint& point, const TPoint& idealPoint) {
		double s = 1.0;
		for (short j = 0; j < OriginalHSS::NumberOfObjectives; j++) {
			s *= (min(point.ObjectiveValues[j], idealPoint.ObjectiveValues[j]) - nadirPoint.ObjectiveValues[j]);
		}

		return s;
	}


	inline short off(short offset, short j) {
		return (j + offset) % OriginalHSS::NumberOfObjectives;
	}


	// Original recursive version of QHV_II (Improved QHV)
	double QHV_II(int start, int end, TPoint& idealPoint, TPoint& nadirPoint, unsigned offset) {
		if (end < start) {
			return 0;
		}

		branches++;
		offset++;
		offset = offset % NumberOfObjectives;

		int oldmaxIndexUsed = maxIndexUsed;

		// if there is just one point
		if (end == start) {
			return volume(nadirPoint, *(indexSetVec[start]), idealPoint);
		}

		// if there are just two points
		if (end - start == 1) {
			double totalVolume = volume(nadirPoint, *(indexSetVec[start]), idealPoint);
			totalVolume += volume(nadirPoint, *(indexSetVec[end]), idealPoint);
			tmpPoint = *(indexSetVec[start]);
			for (short j = 0; j < NumberOfObjectives; j++) {
				tmpPoint.ObjectiveValues[j] = min(tmpPoint.ObjectiveValues[j], indexSetVec[end]->ObjectiveValues[j]);
			}
			totalVolume -= volume(nadirPoint, tmpPoint, idealPoint);
			return totalVolume;
		}

		unsigned iPivot = start;
		double maxVolume = volume(nadirPoint, *(indexSetVec)[iPivot], idealPoint);

		// find the pivot point
		for (unsigned i = start + 1; i <= end; i++) {
			double volumeCurrent = volume(nadirPoint, *(indexSetVec)[i], idealPoint);
			if (maxVolume < volumeCurrent) {
				maxVolume = volumeCurrent;
				iPivot = i;
			}
		}

		double totalVolume = volume(nadirPoint, *(indexSetVec)[iPivot], idealPoint);

		// build subproblems
		unsigned iPos = maxIndexUsed + 1;
		short j;
		short jj;
		int ic = 0;

		TPoint partNadirPoint = nadirPoint;
		TPoint partIdealPoint = idealPoint;

		for (jj = 0; jj < NumberOfObjectives; jj++) {
			j = off(offset, jj);

			if (jj > 0) {
				short j2 = off(offset, jj - 1);
				partIdealPoint.ObjectiveValues[j2] = min(idealPoint.ObjectiveValues[j2], indexSetVec[iPivot]->ObjectiveValues[j2]);
				partNadirPoint.ObjectiveValues[j2] = nadirPoint.ObjectiveValues[j2];
			}

			unsigned partStart = iPos;

			for (unsigned i = start; i <= end; i++) {
				if (i == iPivot)
					continue;

				if (min(idealPoint.ObjectiveValues[j], indexSetVec[i]->ObjectiveValues[j]) >
					min(idealPoint.ObjectiveValues[j], indexSetVec[iPivot]->ObjectiveValues[j])) {
					indexSetVec[iPos++] = indexSetVec[i];
				}
			}

			unsigned partEnd = iPos - 1;
			maxIndexUsed = iPos - 1;

			if (partEnd >= partStart) {
				partNadirPoint.ObjectiveValues[j] = min(idealPoint.ObjectiveValues[j], indexSetVec[iPivot]->ObjectiveValues[j]);
				totalVolume += QHV_II(partStart, partEnd, partIdealPoint, partNadirPoint, offset);
			}
		}

		maxIndexUsed = oldmaxIndexUsed;
		return totalVolume;
	}


	// Original recursive version of QHV_II (Improved QHV)
	double solveQHV_II(vector <TPoint*>& points, TPoint& idealPoint, TPoint& nadirPoint, int numberOfPoints = -1) {
		// allocate memory
		indexSetVec.resize(20000000);
		branches = 0;

		if (numberOfPoints == -1) {
			numberOfPoints = points.size();
		}

		for (int i = 0; i < numberOfPoints; i++) {
			indexSetVec[i] = points[i];
		}

		maxIndexUsed = numberOfPoints - 1;
		double result = QHV_II(0, numberOfPoints - 1, idealPoint, nadirPoint, 0);

		// release memory
		//indexSetVec.clear();
		//indexSetVec.shrink_to_fit();
		return result;
	}
#pragma endregion
	void Load(vector <TPoint*>& points, istream& Stream) {
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

		while (nextChar >= '0' && nextChar <= '9' && points.size() < 100000) {
			i++;
			TPoint* point = new TPoint;

			point->Load(Stream);

			if (Stream.rdstate() == ios::goodbit) {
				points.push_back(point);

				// read the rest of the line
				char c;
				do {
					Stream.get(c);
				} while (c != '\n');
			}
			else
				delete point;

			nextChar = Stream.get();
			Stream.unget();
		}
	}

	void normalize(vector <TPoint*>& points, TPoint& idealPoint, TPoint& nadirPoint) {
		for (unsigned int i = 0; i < points.size(); i++) {
			for (short j = 0; j < NumberOfObjectives; j++) {
				points[i]->ObjectiveValues[j] = (float)((double)points[i]->ObjectiveValues[j] - nadirPoint.ObjectiveValues[j]) /
					(idealPoint.ObjectiveValues[j] - nadirPoint.ObjectiveValues[j]);
			}
		}
	}
	// Calculate contribition to HV of the point using QHV_II
	double getPointContributionQHV_II(int pointIndex, vector <TPoint*>& points, TPoint& idealPoint, TPoint& nadirPoint) {
		vector <TPoint*> tmpPoints = points;
		tmpPoints.erase(tmpPoints.begin() + pointIndex);

		TPoint newIdealPoint;
		for (short j = 0; j < NumberOfObjectives; j++) {
			newIdealPoint.ObjectiveValues[j] = points[pointIndex]->ObjectiveValues[j];
		}

		return volume(nadirPoint, newIdealPoint) - solveQHV_II(tmpPoints, newIdealPoint, nadirPoint, tmpPoints.size());
	}

	// Calculate contribition to HV of the point using QHV_II
	double getPointContributionQHV_II(TPoint* point, vector <TPoint*>& points, TPoint& idealPoint, TPoint& nadirPoint) {
		return getPointContributionQHV_II(find(points.begin(), points.end(), point) - points.begin(), points, idealPoint, nadirPoint);
	}
	void greedyHSSDecLazyQHV_II(vector <TPoint*>& wholeSet, vector <int>& selectedPoints, TPoint& idealPoint, TPoint& nadirPoint, int size_to_get) {
		fstream streamRes("resDec.txt", ios::out | ios::ate | ios::app);

		clock_t t0 = clock();

		priority_queue<BHC, vector<BHC>, compareUBHCMin> queueUBHC;
		double contribution;

		vector <TPoint*> subset = wholeSet;
		bool initialLoop = true;

		unsigned j;
		for (j = 0; j < wholeSet.size(); j++) {
			selectedPoints.push_back(j);
		}

		while (subset.size() > size_to_get) {
			vector<BHC> newLBHC;
			double minContribution = 1e30;
			int minContributor = -1;

			if (initialLoop) {
				for (j = 0; j < wholeSet.size(); j++) {
					contribution = getPointContributionQHV_II(j, subset, idealPoint, nadirPoint);

					if (minContribution > contribution) {
						minContribution = contribution;
						minContributor = j;
					}

					BHC lBHC;
					lBHC.contributor = j;
					lBHC.contribution = contribution;

					queueUBHC.push(lBHC);
				}
				queueUBHC.pop();
				initialLoop = false;
			}
			else {
				double prevContribution = 1;

				while (queueUBHC.size() > 0) {
					BHC lBHC = queueUBHC.top();
					prevContribution = lBHC.contribution;

					if (lBHC.contribution > minContribution) {
						break;
					}

					queueUBHC.pop();
					contribution = getPointContributionQHV_II(wholeSet[lBHC.contributor], subset, idealPoint, nadirPoint);

					if (minContribution > contribution) {
						minContribution = contribution;
						minContributor = lBHC.contributor;
					}

					BHC uBHCTemp;
					uBHCTemp.contributor = lBHC.contributor;
					uBHCTemp.contribution = contribution;

					newLBHC.push_back(uBHCTemp);
				}

				for (auto lBHC : newLBHC) {
					if (lBHC.contributor != minContributor) {
						queueUBHC.push(lBHC);
					}
				}
			}

			//cout << minContributor << ' ';
			int minContributorIndex = find(subset.begin(), subset.end(), wholeSet[minContributor]) - subset.begin();
			subset.erase(subset.begin() + minContributorIndex);
			selectedPoints.erase(selectedPoints.begin() + minContributorIndex);

			if (subset.size() % 10 == 0 || subset.size() < 10) {
				clock_t t0_ = clock();
				double hv = solveQHV_II(subset, idealPoint, nadirPoint);
				t0 += clock() - t0_;

				streamRes << subset.size() << ' ' << clock() - t0 << ' ' << setprecision(10) << hv << endl;
				streamRes.flush();
			}
		}

		cout << endl << endl;
		streamRes << endl;
		streamRes.close();
	}

}

#pragma once
#include "pch.h"
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <set>
#include <ctime>
#include <string>
#include <sstream>
#include <algorithm>
#include <random>
#define MONTE_CARLO_ITERATIONS 100000

namespace OriginalHVE
{


	using namespace std;
	short NumberOfObjectives;
	unsigned long long scalarizingCalls;
	double sumDominatingFactor;
	int maxListSize = 20;
	int maxBranches = 11;
	/** Possible relations between points in multiple objective space */
	enum TCompare { _Dominating, _Dominated, _Nondominated, _EqualSol };

	/** Point in objective space */
	class TPoint {
	public:
		/** Vector of objective values */
		float ObjectiveValues[12];


		/** Copy operator */
		TPoint& operator = (TPoint& point) {
			short i;
			for (i = 0; i < NumberOfObjectives; i++) {
				ObjectiveValues[i] = point.ObjectiveValues[i];
			}
			return *this;
		}


		/** Compares two points in multiple objective space
		*
		*	Returns _Dominated if this is dominated
		*	Returns _Dominating if this is dominating
		*	It may also return _Nondominated or _EqualSol */
		TCompare Compare(TPoint& point) {
			bool bBetter = false;
			bool bWorse = false;

			short i = 0;
			do {
				if (this->ObjectiveValues[i] > point.ObjectiveValues[i])
					bBetter = true;
				if (point.ObjectiveValues[i] > this->ObjectiveValues[i])
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
			}
			return Stream;
		}


		/** Saves objective values to an open Stream
		*
		*	Values are separated by TAB character */
		ostream& Save(ostream& Stream) {
			short i;
			for (i = 0; i < NumberOfObjectives; i++) {
				Stream << ObjectiveValues[i];
				Stream << '\x09';
			}
			Stream << '\x09';
			return Stream;
		}


		double Distance(TPoint& point, TPoint& idealPoint, TPoint& nadirPoint) {
			double s = 0;
			short iobj;
			for (iobj = 0; iobj < NumberOfObjectives; iobj++) {
				double range = idealPoint.ObjectiveValues[iobj] - nadirPoint.ObjectiveValues[iobj];
				if (range == 0)
					range = 1;
				double s1 = (ObjectiveValues[iobj] - point.ObjectiveValues[iobj]) / range;
				s += s1 * s1;
			}
			return sqrt(s);
		}


		double CleanChebycheffScalarizingFunctionOriginal(vector<double>& weightVector, TPoint& referencePoint) {
			scalarizingCalls++;
			double max = -1e30;
			short i;
			for (i = 0; i < NumberOfObjectives; i++) {
				double s = weightVector[i] * (referencePoint.ObjectiveValues[i] - ObjectiveValues[i]);
				if (s > max)
					max = s;
			}

			return max;
		}


		double CleanChebycheffScalarizingFunctionInverse(vector<double>& weightVector, TPoint& referencePoint) {
			scalarizingCalls++;
			double min = 1e30;
			short i;
			for (i = 0; i < NumberOfObjectives; i++) {
				double s = weightVector[i] * (referencePoint.ObjectiveValues[i] - ObjectiveValues[i]);
				if (s < min)
					min = s;
			}

			return min;
		}
	};
	TPoint IdealPoint;
	TPoint NadirPoint;
	TPoint referencePoint;
	class BestSolution {
	public:
		TPoint* solution;
	};

	/** Implementation of nondominated set based on list
	 *
	 * This is simple but not the most efficient solution of nondominated set.
	 **/
	template <class TProblemSolution> class TListSet : public vector <TProblemSolution*>
	{
	protected:
		int				iSetSize;


		/** Update nadir point values
		 *
		 * Method updates nadir point approximation.
		 * Nadir point is approximated on current nondominated set.
		 *
		 * @param iRemovedSolution index of solution to be removed
		 **/
		void UpdateNadir(int iRemovedSolution) {
			short iobj;
			for (iobj = 0; iobj < NumberOfObjectives; iobj++) {
				if ((*this)[iRemovedSolution]->ObjectiveValues[iobj] == ApproximateNadirPoint.ObjectiveValues[iobj]) {
					bool bFirst = true;
					unsigned int i;
					for (i = 0; i < this->size(); i++) {
						if (i != iRemovedSolution) {
							if (bFirst) {
								ApproximateNadirPoint.ObjectiveValues[iobj] = (*this)[i]->ObjectiveValues[iobj];
								bFirst = false;
							}
							else {
								if (ApproximateNadirPoint.ObjectiveValues[iobj] > (*this)[i]->ObjectiveValues[iobj])
									ApproximateNadirPoint.ObjectiveValues[iobj] = (*this)[i]->ObjectiveValues[iobj];
								if (ApproximateNadirPoint.ObjectiveValues[iobj] < (*this)[i]->ObjectiveValues[iobj])
									ApproximateNadirPoint.ObjectiveValues[iobj] = (*this)[i]->ObjectiveValues[iobj];
							}
						}
					}
				}
			}
		}

	public:
		vector <TProblemSolution*> idealDefiningSols;

		bool idealUpdated;
		bool nadirUpdated;

		TPoint ApproximateIdealPoint;

		TPoint ApproximateNadirPoint;

		bool wasEqual;
		bool wasDominated;
		bool wasDominating;

		bool useSortedList = false;

		long updates = 0;

		long dominating = 0;
		long dominated = 0;
		long nondominated = 0;
		long equal = 0;

		/** Update set using given solution
		 *
		 * This function reduce nondominated set to given number of solutions.
		 *
		 * @param Solution possibly nondominated solution
		 * @return if true solution is nondominated and set has been updated with this solution, false solution is dominated by solution in set
		 **/
		virtual bool Update(TPoint& Solution)
		{
			equal = false;
			wasEqual = false;
			wasDominated = false;
			wasDominating = false;

			idealUpdated = false;
			nadirUpdated = false;

			updates++;

			bool bEqual, bDominated, bDominating;

			bool bAdded = false;

			if (this->size() == 0) {
				bAdded = true;
				TProblemSolution* pPattern = (TProblemSolution*)&Solution;
				TProblemSolution* pNewSolution = new TProblemSolution(*pPattern);
				this->push_back(pNewSolution);
				UpdateIdealNadir();
			}
			else {
				bEqual = bDominated = bDominating = false;

				unsigned int i;
				for (i = 0; (i < this->size()) && !bEqual && !bDominated; i++) {

					TCompare ComparisonResult = Solution.Compare(*(*this)[i]);

					switch (ComparisonResult) {
					case _Dominating:
						delete (*this)[i];
						this->erase(this->begin() + i);
						i--;
						bDominating = true;
						break;
					case _Dominated:
					{
						bDominated = true;
					}
					break;
					case _Nondominated:
						break;
					case _EqualSol:
						bEqual = true;
						break;
					}
				}

				if (bDominated && bDominating) {
					// Exception
					cout << Solution.ObjectiveValues[0] << "  " << Solution.ObjectiveValues[1] << "  ";
					cout << "Exception\n";
					cout << "void TListSet::Update (TPoint& Point)\n";
					cout << "bDominated && bDominating\n";
					exit(0);
				}

				if (!bDominated && !bEqual) {
					TProblemSolution* pPattern = (TProblemSolution*)&Solution;
					TProblemSolution* pNewSolution = new TProblemSolution(*pPattern);
					this->push_back(pNewSolution);
					bAdded = true;
				}
				if (bDominated) {
					wasDominated = true;
					dominated++;
				}
				else if (bDominating)
					dominating++;
				else if (bEqual) {
					equal++;
					wasEqual = true;
				}
				else
					nondominated++;

			}

			iSetSize = this->size();

			return bAdded;
		}


		void UpdateIdealNadir()
		{
			// For all points
			unsigned int i;
			for (i = 0; i < this->size(); i++) {
				if ((*this)[i] == NULL) {
					continue;
				}

				// If first point
				if (i == 0) {
					short iobj;
					for (iobj = 0; iobj < NumberOfObjectives; iobj++) {
						ApproximateIdealPoint.ObjectiveValues[iobj] = (*this)[i]->ObjectiveValues[iobj];
						idealDefiningSols[iobj] = (*this)[i];
						ApproximateNadirPoint.ObjectiveValues[iobj] = (*this)[i]->ObjectiveValues[iobj];
					}
				}
				else {
					short iobj;
					for (iobj = 0; iobj < NumberOfObjectives; iobj++) {
						if (ApproximateIdealPoint.ObjectiveValues[iobj] < (*this)[i]->ObjectiveValues[iobj]) {
							ApproximateIdealPoint.ObjectiveValues[iobj] = (*this)[i]->ObjectiveValues[iobj];
							idealDefiningSols[iobj] = (*this)[i];
						}
						else {
							if (ApproximateIdealPoint.ObjectiveValues[iobj] == (*this)[i]->ObjectiveValues[iobj]) {
								idealDefiningSols[iobj] = NULL;
							}
						}
						if (ApproximateNadirPoint.ObjectiveValues[iobj] > (*this)[i]->ObjectiveValues[iobj])
							ApproximateNadirPoint.ObjectiveValues[iobj] = (*this)[i]->ObjectiveValues[iobj];
					}
				}
			}
		}


		void UpdateIdealNadir(TPoint* solution)
		{
			short iobj;
			for (iobj = 0; iobj < NumberOfObjectives; iobj++) {
				if (ApproximateIdealPoint.ObjectiveValues[iobj] < solution->ObjectiveValues[iobj]) {
					ApproximateIdealPoint.ObjectiveValues[iobj] = solution->ObjectiveValues[iobj];
					idealUpdated = true;
					idealDefiningSols[iobj] = solution;
				}
				else {
					if (ApproximateIdealPoint.ObjectiveValues[iobj] == solution->ObjectiveValues[iobj]) {
						idealDefiningSols[iobj] = NULL;
					}

				}
				if ((ApproximateNadirPoint.ObjectiveValues[iobj] > solution->ObjectiveValues[iobj]) || (this->size() == 0)) {
					ApproximateNadirPoint.ObjectiveValues[iobj] = solution->ObjectiveValues[iobj];
					nadirUpdated = true;
				}
			}
		}


		virtual void Add(TPoint& solution)
		{
			idealUpdated = false;
			nadirUpdated = false;

			{
				TProblemSolution* pPattern = (TProblemSolution*)&solution;
				TProblemSolution* pNewSolution = new TProblemSolution(*pPattern);
				UpdateIdealNadir(pNewSolution);
				this->push_back(pNewSolution);
			}
		}


		virtual bool isDominated(TPoint& solution)
		{
			bool bEqual, bDominated, bDominating;

			if (this->size() == 0) {
				return false;
			}
			else {
				bEqual = bDominated = bDominating = false;

				unsigned int i;
				for (i = 0; i < this->size(); i++) {
					TCompare comparisonResult = solution.Compare(*(*this)[i]);

					switch (comparisonResult) {
					case _Dominating:
						return false;
						break;
					case _Dominated:
					{
						return true;
					}
					break;
					case _Nondominated:
						break;
					case _EqualSol:
						return true;
						break;
					}
				}
			}

			return false;
		}


		virtual bool checkUpdate(TPoint& solution)
		{
			equal = false;
			wasEqual = false;
			wasDominating = false;
			wasDominated = false;

			bool bEqual, bDominated, bDominating;

			bool bAdded = false;

			if (this->size() == 0) {
				bAdded = true;
			}
			else {
				bEqual = bDominated = bDominating = false;

				unsigned int i;
				for (i = 0; (i < this->size()) && !bEqual && !bDominated; i++) {

					TCompare comparisonResult = solution.Compare(*(*this)[i]);

					switch (comparisonResult) {
					case _Dominating:
						UpdateNadir(i);
						delete (*this)[i];
						this->erase(this->begin() + i);
						i--;
						wasDominating = true;
						bDominating = true;
						break;
					case _Dominated:
					{
						wasDominated = true;
						bDominated = true;
					}
					break;
					case _Nondominated:
						break;
					case _EqualSol:
						bEqual = true;
						break;
					}
				}

				if (bDominated && bDominating) {
					// Exception
					cout << solution.ObjectiveValues[0] << "  " << solution.ObjectiveValues[1] << "  ";
					cout << "Exception\n";
					cout << "void TListSet::Update (TPoint& Point)\n";
					cout << "bDominated && bDominating\n";
					exit(0);
				}

				if (!bDominated && !bEqual) {
					bAdded = true;
				}
				if (bEqual) {
					wasEqual = true;
				}
			}

			return bAdded;
		}


		/** Delete all solutions from set.
		*
		* Every solution in set is released and vector is reallocated to size 0.
		**/
		virtual void DeleteAll()
		{
			//		TSolutionsSet::DeleteAll();

			iSetSize = 0;
		}


		/** This function choose random solution from set of solutions
		 *
		 * Probability of choose for every solution should be equal.
		 *
		 * @param pSolution reference to pointer where solution will be placed
		 **/
		virtual void GetRandomSolution(TPoint*& pSolution)
		{
			int	iIndex = 0;

			// old code
			if (iSetSize <= 0) {
				pSolution = NULL;
			}
			else {
				iIndex = rand() % iSetSize;
				pSolution = (TPoint*)(*this)[iIndex];
			}
		}


		/** Destruct object
		 *
		 **/
		TListSet() {
			idealDefiningSols.resize(NumberOfObjectives, NULL);
		}

		/*	~TListSet () {
				DeleteAll();
			};*/
	};

	template <class TProblemSolution> class TComponent {
	public:
		TListSet<TProblemSolution> listSet;
	public:

		TPoint approximateIdealPoint;
		TPoint approximateNadirPoint;

		double scalarizingFunctionValue;

		vector <TComponent<TProblemSolution>*> branches;

		TComponent <TProblemSolution>* parent = NULL;

		long numberOfSolutions() {
			if (listSet.size() > 0)
				return listSet.size();
			else {
				long sizeValue = 0;
				unsigned i;
				for (i = 0; i < branches.size(); i++)
					sizeValue += branches[i]->numberOfSolutions();
				return sizeValue;
			}
		}

		long numberOfNodes() {
			long sizeValue = 0;
			if (listSet.size() == 0)
				sizeValue = 1;
			unsigned i;
			for (i = 0; i < branches.size(); i++)
				sizeValue += branches[i]->numberOfNodes();
			return sizeValue;
		}

		long numberOfLeafs() {
			long sizeValue = 0;
			if (listSet.size() != 0)
				sizeValue = 1;
			unsigned i;
			for (i = 0; i < branches.size(); i++)
				sizeValue += branches[i]->numberOfNodes();
			return sizeValue;
		}

		// Test method only
		//bool checkIdeals() {
		//	if (ComponentSolution != NULL) {
		//		return true;
		//	}
		//	else {
		//		bool correct = true;
		//		unsigned j;
		//		for (j = 0; j < branches.size(); j++) {
		//			TCompare ComparisonResult = approximateIdealPoint.Compare(branches[j]->approximateIdealPoint);
		//			correct = correct && (ComparisonResult == _Dominating || ComparisonResult == _EqualSol);
		//			if (!correct)
		//				return false;
		//			correct = correct && branches[j]->checkIdeals();
		//			if (!correct)
		//				return false;
		//		}
		//		return correct;
		//	}
		//}

		void print(int level, fstream& stream) {

			string s;
			unsigned i;
			for (i = 0; i < level; i++)
				s += ' ';
			stream << s;
			unsigned j;
			stream << s << "Ideal ";

			for (j = 0; j < NumberOfObjectives; j++) {
				stream << approximateIdealPoint.ObjectiveValues[j] << ' ';
				if (approximateIdealPoint.ObjectiveValues[j] == 0)
					int a = 1;
			}
			stream << '\n';
			stream << s << "Nadir ";
			for (j = 0; j < NumberOfObjectives; j++) {
				stream << approximateNadirPoint.ObjectiveValues[j] << ' ';
				if (approximateNadirPoint.ObjectiveValues[j] == 0)
					int a = 1;
			}
			stream << '\n';

			if (listSet.size() > 0) {
				unsigned k;
				for (k = 0; k < listSet.size(); k++) {
					stream << s << " Leaf ";
					unsigned j;
					for (j = 0; j < NumberOfObjectives; j++)
						stream << listSet[k]->ObjectiveValues[j] << ' ';
					stream << '\n';
				}
			}
			else {
				for (j = 0; j < branches.size(); j++)
					branches[j]->print(level + 1, stream);
			}
		}

		//		void updateIdeal() {
		//			unsigned i;
		//			for (i = 0; i < branches.size(); i++) {
		//				if (branches[i]->listSet.size() == 0 && branches[i]->branches.size() == 0)
		//					continue;
		//				if (branches[i]->approximateIdealPoint.ObjectiveValues[0] == 0)
		//					int a = 1;
		//				if (i == 0) {
		//					approximateIdealPoint.ObjectiveValues = branches[i]->approximateIdealPoint.ObjectiveValues;
		//				}
		//				else {
		//					unsigned j;
		//					for (j = 0; j < NumberOfObjectives; j++) {
		//#ifdef MFPC
		//						fpc++;
		//#endif // MFPC
		//						if (Objectives[j].ObjectiveType == _Max) {
		//							if (approximateIdealPoint.ObjectiveValues[j] > branches[i]->approximateIdealPoint.ObjectiveValues[j]) {
		//								approximateIdealPoint.ObjectiveValues[j] = branches[i]->approximateIdealPoint.ObjectiveValues[j];
		//							}
		//#ifdef MFPC
		//							fpc++;
		//#endif // MFPC
		//						}
		//						else {
		//#ifdef MFPC
		//							fpc++;
		//#endif // MFPC
		//							if (approximateIdealPoint.ObjectiveValues[j] < branches[i]->approximateIdealPoint.ObjectiveValues[j]) {
		//								approximateIdealPoint.ObjectiveValues[j] = branches[i]->approximateIdealPoint.ObjectiveValues[j];
		//							}
		//#ifdef MFPC
		//							fpc++;
		//#endif // MFPC
		//						}
		//					}
		//				}
		//			}
		//			if (parent != NULL)
		//				parent->updateIdeal();
		//		}
		void print(int tab)
		{
			for (int i = 0; i < tab; i++)
				std::cout << "  ";
			std::cout << branches.size() << '\n';
			for (int i = 0; i < branches.size(); i++) {
				branches[i]->print(tab + 1);
			}
		}
		void minScalarizingFunction(double& currentMin, TPoint& ReferencePoint,
			vector <double>& WeightVector, BestSolution& bestSolution) {

			if (this->listSet.size() > 0) {
				double MinScalarizingFunctionValue = 1e30;
				unsigned int i;
				for (i = 0; i < this->listSet.size(); i++) {
					double ScalarizingFunctionValue;

					ScalarizingFunctionValue = listSet[i]->CleanChebycheffScalarizingFunctionOriginal(WeightVector, ReferencePoint);

					if (ScalarizingFunctionValue < currentMin) {
						currentMin = ScalarizingFunctionValue;
						bestSolution.solution = listSet[i];
					}
				}
			}
			else {
				// Find the best branch
				int iBest = -1;
				double MinScalarizingFunctionValue = 1e30;
				unsigned j;
				for (j = 0; j < branches.size(); j++) {
					double ScalarizingFunctionValue;

					ScalarizingFunctionValue = branches[j]->approximateIdealPoint.CleanChebycheffScalarizingFunctionOriginal(WeightVector, ReferencePoint);

					if (ScalarizingFunctionValue < MinScalarizingFunctionValue) {
						MinScalarizingFunctionValue = ScalarizingFunctionValue;
						iBest = j;
					}
				}
				if (MinScalarizingFunctionValue < currentMin) {
					branches[iBest]->minScalarizingFunction(currentMin, ReferencePoint,
						WeightVector, bestSolution);
				}

				// Check other branches
				for (j = 0; j < branches.size(); j++) {
					if (j == iBest)
						continue;

					double ScalarizingFunctionValue;

					ScalarizingFunctionValue = branches[j]->approximateIdealPoint.CleanChebycheffScalarizingFunctionOriginal(WeightVector, ReferencePoint);

					if (ScalarizingFunctionValue < currentMin) {
						branches[j]->minScalarizingFunction(currentMin, ReferencePoint,
							WeightVector, bestSolution);
					}
				}
			}

		}

		void maxScalarizingFunction(double& currentMax, TPoint& ReferencePoint,
			vector <double>& WeightVector, BestSolution& bestSolution) {

			if (this->listSet.size() > 0) {
				double MaxScalarizingFunctionValue = -11e30;
				unsigned int i;
				for (i = 0; i < this->listSet.size(); i++) {

					double ScalarizingFunctionValue;

					ScalarizingFunctionValue = listSet[i]->CleanChebycheffScalarizingFunctionInverse(WeightVector, ReferencePoint);

					if (ScalarizingFunctionValue > currentMax) {
						currentMax = ScalarizingFunctionValue;
						bestSolution.solution = listSet[i];
					}
				}
			}
			else {
				// Find the best branch
				int iBest = -1;
				double MaxScalarizingFunctionValue = -1e30;
				unsigned j;

				for (j = 0; j < branches.size(); j++) {

					double ScalarizingFunctionValue =
						branches[j]->approximateIdealPoint.CleanChebycheffScalarizingFunctionInverse(WeightVector, ReferencePoint);

					branches[j]->scalarizingFunctionValue = ScalarizingFunctionValue;

					if (ScalarizingFunctionValue > MaxScalarizingFunctionValue) {
						MaxScalarizingFunctionValue = ScalarizingFunctionValue;
						iBest = j;
					}
				}
				if (MaxScalarizingFunctionValue > currentMax) {
					branches[iBest]->maxScalarizingFunction(currentMax, ReferencePoint, WeightVector, bestSolution);
				}

				// Check other branches
				for (j = 0; j < branches.size(); j++) {
					if (j == iBest)
						continue;

					double ScalarizingFunctionValue = branches[j]->scalarizingFunctionValue;
					if (ScalarizingFunctionValue > currentMax) {
						branches[j]->maxScalarizingFunction(currentMax, ReferencePoint, WeightVector, bestSolution);
					}
				}
			}

		}

		void updateIdeal(TComponent <TProblemSolution>* node) {
			if (node->listSet.size() == 0 && node->branches.size() == 0)
				return;
			bool changed = false;
			unsigned j;
			for (j = 0; j < NumberOfObjectives; j++) {
#ifdef MFPC
				fpc++;
#endif // MFPC
				if (approximateIdealPoint.ObjectiveValues[j] < node->approximateIdealPoint.ObjectiveValues[j]) {
					approximateIdealPoint.ObjectiveValues[j] = node->approximateIdealPoint.ObjectiveValues[j];
					changed = true;
				}
#ifdef MFPC
				fpc++;
#endif // MFPC
				if (approximateNadirPoint.ObjectiveValues[j] > node->approximateNadirPoint.ObjectiveValues[j]) {
					approximateNadirPoint.ObjectiveValues[j] = node->approximateNadirPoint.ObjectiveValues[j];
					changed = true;
				}
			}
			if (changed && parent != NULL)
				parent->updateIdeal(this);
		}

		double insertDistance(TPoint& Solution, bool useOrginalNDTree) {
			double s = 0;

			// Lexycographic criterion
			if (!useOrginalNDTree && this->approximateIdealPoint.Compare(Solution) == _Dominating) {
				s += -10;
			}

			int iobj; for (iobj = 0; iobj < NumberOfObjectives; iobj++) {
				double center;

				if (useOrginalNDTree) {
					center = (approximateIdealPoint.ObjectiveValues[iobj] + approximateNadirPoint.ObjectiveValues[iobj]) / 2.0;
				}
				else {
					center = approximateIdealPoint.ObjectiveValues[iobj];
				}

				double s1 = (center - Solution.ObjectiveValues[iobj]); // / Range;
				s += s1 * s1;
			}

			return s;
		}

		void splitByClustering(bool useOrginalNDTree) {
			int numberOfClusert = min(maxListSize + 1, maxBranches);//8

			vector <TProblemSolution*> seeds;

			int j;
			for (j = 0; j < numberOfClusert; j++) {
				branches.push_back(new TComponent(this));
				if (j == 0) {
					// Find solution furthest from all other solutions
					double maxDistance = 0;
					int maxIndex = 0;
					unsigned i;
					for (i = 0; i < listSet.size(); i++) {
						unsigned i2;
						double sumDistance = 0;
						for (i2 = 1; i2 < listSet.size(); i2++) {
							if (i != i2)
								sumDistance += listSet[i]->Distance(*(listSet[i2]), IdealPoint, NadirPoint);
							//							sumDistance += insertDistance3 (*(listSet[i]), *(listSet[i2]), referencePoint);
						}
						if (i == 0) {
							maxDistance = sumDistance;
						}
						else {
#ifdef MFPC
							fpc++;
#endif // MFPC
							if (maxDistance < sumDistance) {
								maxDistance = sumDistance;
								maxIndex = i;
							}
						}
					}
					TProblemSolution* solution = (TProblemSolution*)(listSet[maxIndex]);
					branches[j]->add(*solution);
					delete listSet[maxIndex];
					listSet.erase(listSet.begin() + maxIndex);
				}
				else {
					// Find solution furthest from all other clusters
					double maxDistance;
					int maxIndex = 0;
					unsigned i;
					for (i = 0; i < listSet.size(); i++) {
						unsigned i2;
						double sumDistance = 0;
						for (i2 = 0; i2 < seeds.size(); i2++) {
							if (i != i2)
								sumDistance += listSet[i]->Distance(*(seeds[i2]), IdealPoint, NadirPoint);
							//							sumDistance += insertDistance3(*(listSet[i]), *(seeds[i2]), referencePoint);
						}
						if (i == 0) {
							maxDistance = sumDistance;
						}
						else {
#ifdef MFPC
							fpc++;
#endif // MFPC
							if (maxDistance < sumDistance) {
								maxDistance = sumDistance;
								maxIndex = i;
							}
						}
					}
					TProblemSolution* solution = (TProblemSolution*)(listSet[maxIndex]);
					branches[j]->add(*solution);
					delete listSet[maxIndex];
					listSet.erase(listSet.begin() + maxIndex);
				}
			}

			// Now add each solution to closest branch
			unsigned i;
			for (i = 0; i < listSet.size(); i++) {
				TProblemSolution* Solution = (TProblemSolution*)listSet[i];
				//			double minDistance = branches[0]->insertDistance2(*Solution, referencePoint);
				double minDistance = branches[0]->insertDistance(*Solution, useOrginalNDTree);
				int minIndex = 0;
				for (j = 1; j < branches.size(); j++) {
					//				double distanceValue = branches[j]->insertDistance2(*Solution, referencePoint);
					double distanceValue = branches[j]->insertDistance(*Solution, useOrginalNDTree);
#ifdef MFPC
					fpc++;
#endif // MFPC
					if (minDistance > distanceValue) {
						minDistance = distanceValue;
						minIndex = j;
					}
				}
				branches[minIndex]->insert(*Solution, useOrginalNDTree);
			}
			listSet.DeleteAll();
			listSet.clear();

		}

		double range(TListSet<TProblemSolution>& listSet, int iobj) {
			double min = 1e30;
			double max = -1e30;
			for (auto p : listSet) {
				if (min > p->ObjectiveValues[iobj]) {
					min = p->ObjectiveValues[iobj];
				}
				if (max < p->ObjectiveValues[iobj]) {
					max = p->ObjectiveValues[iobj];
				}
			}
			return max - min;
		}

		//void splitByKDD() {
		//	branches.push_back(new TComponent(this));
		//	branches.push_back(new TComponent(this));
		//	branches.push_back(new TComponent(this));
		//	branches.push_back(new TComponent(this));
		//	branches.push_back(new TComponent(this));
		//	branches.push_back(new TComponent(this));
		//	branches.push_back(new TComponent(this));
		//	branches.push_back(new TComponent(this));

		//	int so1 = -1;
		//	int so2 = -1;
		//	int so3 = -1;
		//	double max = -1e30;

		//	int iobj;
		//	for (iobj = 0; iobj < NumberOfObjectives; iobj++) {
		//		double r = range(listSet, iobj);
		//		if (max < r) {
		//			so3 = so2;
		//			so2 = so1;
		//			so1 = iobj;
		//			max = r;
		//		}
		//	}

		//	splitObjective = so1;

		//	std::sort(listSet.begin(), listSet.end(), kDDCompare);

		//	//		splitObjective = (splitObjective + 1) % NumberOfObjectives;

		//	//		int so2 = splitObjective;

		//	double midValue = listSet[listSet.size() / 2]->ObjectiveValues[splitObjective];

		//	splitObjective = so2;

		//	std::sort(listSet.begin(), listSet.end(), kDDCompare);

		//	//		splitObjective = (splitObjective + 1) % NumberOfObjectives;

		//	double midValue2 = listSet[listSet.size() / 2]->ObjectiveValues[splitObjective];

		//	splitObjective = so3;

		//	std::sort(listSet.begin(), listSet.end(), kDDCompare);

		//	double midValue3 = listSet[listSet.size() / 2]->ObjectiveValues[splitObjective];

		//	int i;
		//	for (i = 0; i < listSet.size(); i++) {
		//		TProblemSolution* Solution = (TProblemSolution*)listSet[i];
		//		int iBranch = -1;
		//		if (Solution->ObjectiveValues[so1] < midValue) {
		//			if (Solution->ObjectiveValues[so2] < midValue2) {
		//				if (Solution->ObjectiveValues[so3] < midValue2) {
		//					iBranch = 0;
		//				}
		//				else {
		//					iBranch = 1;
		//				}
		//			}
		//			if (Solution->ObjectiveValues[so3] < midValue2) {
		//				iBranch = 2;
		//			}
		//			else {
		//				iBranch = 3;
		//			}
		//		}
		//		else {
		//			if (Solution->ObjectiveValues[so2] < midValue2) {
		//				if (Solution->ObjectiveValues[so3] < midValue2) {
		//					iBranch = 4;
		//				}
		//				else {
		//					iBranch = 5;
		//				}
		//			}
		//			if (Solution->ObjectiveValues[so3] < midValue2) {
		//				iBranch = 6;
		//			}
		//			else {
		//				iBranch = 7;
		//			}
		//		}
		//		if (branches[iBranch]->listSet.size() == 0) {
		//			branches[iBranch]->add(*Solution);
		//		}
		//		else {
		//			branches[iBranch]->insert(*Solution);
		//		}
		//	}
		//	listSet.DeleteAll();
		//	listSet.clear();
		//}

		/*	void splitByClustering2() {
				vector <vector <double>> characteristicDirections;

				vector <double> middleDirection;
				middleDirection.resize(NumberOfObjectives, 0);

				int j;
				for (j = 0; j < NumberOfObjectives; j++) {
					double max = -1e30;
					int iMax = -1;
					unsigned i;
					for (i = 0; i < listSet.size(); i++) {
						if (max < listSet[i]->ObjectiveValues[j]) {
							max = listSet[i]->ObjectiveValues[j];
							iMax = j;
						}
					}
					vector <double> characteristicDirection;
					characteristicDirection.resize(NumberOfObjectives);
					int j2;
					for (j2 = 0; j2 < NumberOfObjectives; j2++) {
						characteristicDirection[j2] = referencePoint.ObjectiveValues[j2] - listSet[iMax]->ObjectiveValues[j2];
					}

					if (find(characteristicDirections.begin(), characteristicDirections.end(), characteristicDirection) ==
						characteristicDirections.end()) {
						characteristicDirections.push_back(characteristicDirection);

						branches.push_back(new TComponent(this));
						TProblemSolution* solution = (TProblemSolution*)(listSet[iMax]);
						branches[j]->add(*solution);
						delete listSet[iMax];
						listSet.erase(listSet.begin() + iMax);
					}

				}

				for (j2 = 0; j2 < NumberOfObjectives; j2++) {
					middleDirection[j2] += characteristicDirections[;
				}

				int i;
				for (i = 0; i < listSet.size(); i++) {
					vector <double> pointDirection;
					pointDirection.resize(NumberOfObjectives);
					int j;
					for (j = 0; j < NumberOfObjectives; j++) {
						pointDirection[j] = referencePoint.ObjectiveValues[j] - listSet[i]->ObjectiveValues[j];
					}

					double min = 1e30;
					int i2Min = -1;
					int i2;
					for (i2 = 0; i2 < branches.size(); i2) {
						double d = dotProduct(characteristicDirections[i2], pointDirection);
						if (min > d) {
							min = d;
							i2Min = i2;
						}
					}
					TProblemSolution* Solution = (TProblemSolution*)listSet[i];
					branches[i2Min]->insert(*Solution);
				}
				listSet.DeleteAll();
				listSet.clear();
			}
			*/
		void add(TPoint& Solution) {
			listSet.Update(Solution);
			approximateIdealPoint = Solution;
			approximateNadirPoint = Solution;
		}

		void insert(TPoint& Solution, bool useOrginalNDTree) {
			if (listSet.size() > 0) {
				listSet.Add(Solution);
				if (listSet.idealUpdated) {
					approximateIdealPoint = listSet.ApproximateIdealPoint;
					if (parent != NULL) {
						parent->updateIdeal(this);
					}
				}
				if (listSet.size() > maxListSize) {
					splitByClustering(useOrginalNDTree);
					//				splitByKDD();
				}
			}
			else {
				if (branches.size() == 0) {
					add(Solution);
				}
				else {
					// Find closest branch
					double minDistance = branches[0]->insertDistance(Solution, useOrginalNDTree);
					int minIndex = 0;

					unsigned i;
					for (i = 1; i < branches.size(); i++) {
						double distanceValue = branches[i]->insertDistance(Solution, useOrginalNDTree);
#ifdef MFPC
						fpc++;
#endif // MFPC
						if (minDistance > distanceValue) {
							minDistance = distanceValue;
							minIndex = i;
						}
					}
					branches[minIndex]->insert(Solution, useOrginalNDTree);
				}
			}
		}

		virtual void Update(TPoint& Solution, bool& nondominated, bool& dominated, bool& dominating, bool& equal,
			bool& added, bool& toInsert) {

			// Compare to approximateIdealPoint
			TCompare NadirComparisonResult = Solution.Compare(approximateNadirPoint);

			if (NadirComparisonResult == _Dominated || NadirComparisonResult == _EqualSol) {
				dominated = true;
				nondominated = false;
				return;
			}

			TCompare IdealComparisonResult = Solution.Compare(approximateIdealPoint);

			if (IdealComparisonResult == _Dominating || IdealComparisonResult == _EqualSol) {
				// if a leaf
				if (listSet.size() > 0) {
					listSet.DeleteAll();
				}
				else {
					int k;
					for (k = 0; k < branches.size(); k++)
						delete branches[k];
					branches.clear();
				}
				dominating = true;
				nondominated = false;
				toInsert = true;
				added = false;
				return;
			}
			else if (IdealComparisonResult == _Dominated || NadirComparisonResult == _Dominating) {
				// if a leaf
				if (listSet.size() > 0) {
					added = listSet.checkUpdate(Solution);
					if (listSet.wasDominated) {
						nondominated = false;
						dominated = true;
						return;
					}
					if (listSet.wasEqual) {
						nondominated = false;
						equal = true;
						return;
					}
					if (added) {
						added = false;
						if (listSet.wasDominating) {
							toInsert = true;
						}
						return;
					}
					else {
						nondominated = false;
						dominated = true;
						return;
					}
				}
				else {
					unsigned i;
					for (i = 0; i < branches.size() && !dominated && !added && !equal; i++) {
						branches[i]->Update(Solution, nondominated, dominated, dominating, equal,
							added, toInsert);
						if (branches[i]->listSet.size() == 0 && branches[i]->branches.size() == 0) {
							delete branches[i];
							branches.erase(branches.begin() + i);
							i--;
						}
					}
					return;
				}
			}
			// outside the box
			else {
				toInsert = true;
			}
		}

		virtual bool isDominated(TPoint& Solution) {

			// Compare to approximateIdealPoint
			TCompare NadirComparisonResult = Solution.Compare(approximateNadirPoint);

			if (NadirComparisonResult == _Dominated || NadirComparisonResult == _EqualSol) {
				return true;
			}

			TCompare IdealComparisonResult = Solution.Compare(approximateIdealPoint);

			if (IdealComparisonResult == _Dominating || IdealComparisonResult == _EqualSol) {
				return false;
			}
			else if (IdealComparisonResult == _Dominated || NadirComparisonResult == _Dominating) {
				// if a leaf
				if (listSet.size() > 0) {
					return listSet.isDominated(Solution);
				}
				else {
					bool wasDominated = false;

					unsigned i;
					for (i = 0; i < branches.size() && !wasDominated; i++) {
						wasDominated = branches[i]->isDominated(Solution);
					}
					return wasDominated;
				}
			}
			// outside the box
			else {
				return false;
			}
		}

		ostream& Save(ostream& Stream) {
			unsigned i;
			if (listSet.size() > 0) {
				unsigned i;
				for (i = 0; i < listSet.size(); i++) {
					listSet[i]->Save(Stream);
					Stream << '\n';
				}
			}
			else {
				unsigned j;
				for (j = 0; j < branches.size(); j++)
					branches[j]->Save(Stream);
			}

			return Stream;
		}

		TComponent(TComponent <TProblemSolution>* parentParam) {
			this->parent = parentParam;
		}

		~TComponent() {
			for (auto s : listSet) {
				delete s;
			}
			//		listSet.DeleteAll();
			listSet.clear();
			int k;
			for (k = 0; k < branches.size(); k++)
				delete branches[k];
			branches.clear();
		}
	};
	template <class TProblemSolution> class TTreeSet
	{
	protected:
	public:
		vector <TProblemSolution*> listSet;

		TComponent<TProblemSolution>* root = NULL;

		void saveToList() {
			unsigned i;
			for (i = 0; i < listSet.size(); i++)
				delete listSet[i];
			listSet.clear();
			root->saveToList(listSet, false);
		}

		long numberOfSolutions() {
			return root->numberOfSolutions();
		}

		virtual bool isDominated(TPoint& Solution) {
			return root->isDominated(Solution);
		}

		virtual bool Update(TPoint& Solution, bool checkDominance, bool useOrginalNDTree = false) {

			bool nondominated = true;
			bool dominated = false;
			bool dominating = false;
			bool added = false;
			bool toInsert = false;
			bool equal = false;
			bool result = false;

			if (root == NULL) {
				root = new TComponent <TProblemSolution>(NULL);
				root->add(Solution);
				result = true;
			}
			else {
				if (checkDominance) {
					root->Update(Solution, nondominated, dominated, dominating, equal,
						added, toInsert);
				}
				if (!checkDominance || (nondominated && toInsert) || (dominating && toInsert)) {
					root->insert(Solution, useOrginalNDTree);
					result = true;
				}
				else if (nondominated) {
					root->insert(Solution, useOrginalNDTree);
					result = true;
				}
			}

			return result;

		}

		void Save(char* FileName) {
			fstream Stream(FileName, ios::out);
			root->Save(Stream);
			Stream.close();
		}


		virtual void DeleteAll() {
			delete root;
		}

		TTreeSet() {
			int j;
			for (j = 0; j < NumberOfObjectives; j++) {
				NadirPoint.ObjectiveValues[j] = 0;
				IdealPoint.ObjectiveValues[j] = 0;
			}
		}

		~TTreeSet() {
			DeleteAll();
		};
	};
	double norm(vector <double>& p) {
		double s = 0;
		int j;
		for (j = 0; j < NumberOfObjectives; j++) {
			s += p[j] * p[j];
		}
		return sqrt(s);
	}
	void normalize(vector <double>& p) {
		double nrm = norm(p);
		double s = 0;
		int j;
		for (j = 0; j < NumberOfObjectives; j++) {
			p[j] /= nrm;
		}
	}


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
	void normalize(vector <TPoint*>& set, TPoint& idealPoint, TPoint& nadirPoint) {
		//Normalize the values to 0-1
		unsigned int i;
		for (i = 0; i < set.size(); i++) {
			if (set[i] == NULL) {
				continue;
			}

			short j;
			for (j = 0; j < NumberOfObjectives; j++) {
				set[i]->ObjectiveValues[j] = ((double)set[i]->ObjectiveValues[j] - nadirPoint.ObjectiveValues[j]) /
					(idealPoint.ObjectiveValues[j] - nadirPoint.ObjectiveValues[j]);
			}
		}
	}
	double approximateHVMax(vector <TPoint*>& set, TPoint& idealPoint, TPoint& nadirPoint, int numberOfSolutions,
		bool useOriginalNDTree = false, unsigned seed = 0, bool checkDominance = true) {

		std::uniform_real_distribution<double> uniformDouble(0, 1);
		std::normal_distribution<double> normal(0, 1);
		std::default_random_engine randEng;
		if (seed > 0) {
			randEng.seed(seed);
		}

		vector <double> unitPoint;
		unitPoint.resize(NumberOfObjectives, 1);

		vector <double> randomDirectionVector;
		randomDirectionVector.resize(NumberOfObjectives);

		vector <double> randomWeightsVector;
		randomWeightsVector.resize(NumberOfObjectives);

		referencePoint = idealPoint;

		// Put all points in ND-Tree
		TTreeSet<TPoint> treeSet;

		int ii;
		for (ii = 0; ii < numberOfSolutions; ii++) {
			treeSet.Update(*set[ii], checkDominance, useOriginalNDTree);
			//if (ii < 300 && ii % 100 == 0)
			//{
			//	treeSet.root->print(0);
			//	std::cout << "-------------------------\n";
			//}
		}
		//treeSet.root->print(0);
		//cout << treeSet.numberOfSolutions() << '\n';

		double sumDominating = 0;

		int i;
		for (i = 0; i < MONTE_CARLO_ITERATIONS; i++) {
			// Draw a random direction and corresponding weight vector

			// Draw a random point (drection) on the surface of a ball with radius 1
			short j;
			for (j = 0; j < NumberOfObjectives; j++) {
				randomDirectionVector[j] = normal(randEng);
				if (randomDirectionVector[j] < 0) {
					randomDirectionVector[j] = -randomDirectionVector[j];
				}
				if (randomDirectionVector[j] == 0) {
					randomDirectionVector[j] = 0.00000000001;
				}
			}
			normalize(randomDirectionVector);

			for (j = 0; j < NumberOfObjectives; j++) {
				randomWeightsVector[j] = -1 / randomDirectionVector[j];
			}

			BestSolution bestSolution;
			double currentMax = -1e30;
			treeSet.root->maxScalarizingFunction(currentMax, nadirPoint, randomWeightsVector, bestSolution);
			//scalarizingCalls++;
			// Calculate Euclidean distance to this point
			sumDominating += pow(currentMax, NumberOfObjectives);
		}

		return sumDominating * sumDominatingFactor;
	}

}
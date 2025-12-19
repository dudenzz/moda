
#include "ListSet.h"
#include "DataSet.h"
namespace moda {
	template <class Solution> void ListSet<Solution>::UpdateNadir(int iRemovedSolution) {

		int iobj; for (iobj = 0; iobj < NumberOfObjectives; iobj++) {
			{
				if ((*this)[iRemovedSolution]->ObjectiveValues[iobj] == ApproximateNadirPoint.ObjectiveValues[iobj]) {
					bool bFirst = true;
					unsigned int i; for (i = 0; i < this->size(); i++) {
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

	}
	template <class Solution> bool ListSet<Solution>::insert(Point& _Solution) {
		if (NumberOfObjectives == -1)
			NumberOfObjectives = _Solution.NumberOfObjectives;
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
			Point* pPattern = (Point*)&_Solution;
			Point* pNewSolution = new Point(*pPattern);
			
			this->push_back(pNewSolution);
			Point temp1;
			temp1 = Point(_Solution);
			Point temp2;
			temp2 = Point(_Solution);
			ApproximateIdealPoint = temp1;
			ApproximateNadirPoint = temp2;
			UpdateIdealNadir();
			
		}
		else {
			{
				bEqual = bDominated = bDominating = false;

				unsigned int i;
				for (i = 0; (i < this->size()) && !bEqual && !bDominated; i++) {

					ComparisonResult ComparisonResult = _Solution.Compare(*(*this)[i], this->maximization);

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
					std::cout << _Solution.ObjectiveValues[0] << "  " << _Solution.ObjectiveValues[1] << "  ";
					std::cout << "Exception\n";
					std::cout << "void TListSet<Solution>::Update (TPoint& Point)\n";
					std::cout << "bDominated && bDominating\n";
					exit(0);
				}

				if (!bDominated && !bEqual) {
					Point* pPattern = (Point*)&_Solution;
					Point* pNewSolution = new Point(*pPattern);
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
		}

		iSetSize = this->size();

		return bAdded;
	}
	template <class Solution> void ListSet<Solution>::UpdateIdealNadir() {
		// For all points
		unsigned int i; for (i = 0; i < this->size(); i++) {
			if ((*this)[i] == NULL) {
				continue;
			}

			// If first point
			if (i == 0) {
				int iobj; for (iobj = 0; iobj < NumberOfObjectives; iobj++) {
					ApproximateIdealPoint.ObjectiveValues[iobj] = (*this)[i]->ObjectiveValues[iobj];
					ApproximateNadirPoint.ObjectiveValues[iobj] = (*this)[i]->ObjectiveValues[iobj];
				}
			}
			else {
				int iobj; for (iobj = 0; iobj < NumberOfObjectives; iobj++) {
					{
						if (ApproximateIdealPoint.ObjectiveValues[iobj] < (*this)[i]->ObjectiveValues[iobj])
							ApproximateIdealPoint.ObjectiveValues[iobj] = (*this)[i]->ObjectiveValues[iobj];
						if (ApproximateNadirPoint.ObjectiveValues[iobj] > (*this)[i]->ObjectiveValues[iobj])
							ApproximateNadirPoint.ObjectiveValues[iobj] = (*this)[i]->ObjectiveValues[iobj];
					}
				}
			}
		}
	}	
	template <class Solution> void ListSet<Solution>::UpdateIdealNadir(Point& _Solution) {
		int iobj; for (iobj = 0; iobj < NumberOfObjectives; iobj++) {
			{
				if (ApproximateIdealPoint.ObjectiveValues[iobj] < _Solution.ObjectiveValues[iobj]) {
					ApproximateIdealPoint.ObjectiveValues[iobj] = _Solution.ObjectiveValues[iobj];
					idealUpdated = true;
				}
				if ((ApproximateNadirPoint.ObjectiveValues[iobj] > _Solution.ObjectiveValues[iobj]) || (this->size() == 0)) {
					ApproximateNadirPoint.ObjectiveValues[iobj] = _Solution.ObjectiveValues[iobj];
					nadirUpdated = true;
				}
			}
		}
	}
	template <class Solution> void ListSet<Solution>::Add(Point& _Solution) {
		idealUpdated = false;
		nadirUpdated = false;

		{
			Point* pPattern = (Point*)&_Solution;
			Point* pNewSolution = new Point(*pPattern);
			UpdateIdealNadir(*pNewSolution);
			this->push_back(pNewSolution);
		}
	}
	template <class Solution> bool ListSet<Solution>::isDominated(Point& _Solution) {
		bool bEqual, bDominated, bDominating;

		if (this->size() == 0) {
			return false;
		}
		else {
			bEqual = bDominated = bDominating = false;

			unsigned int i; for (i = 0; i < this->size(); i++) {

				ComparisonResult ComparisonResult = _Solution.Compare(*(*this)[i], maximization);

				switch (ComparisonResult) {
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
	template <class Solution> bool ListSet<Solution>::checkUpdate(Point& _Solution) {

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

			unsigned int i; for (i = 0; (i < this->size()) && !bEqual && !bDominated; i++) {

				ComparisonResult ComparisonResult = _Solution.Compare(*(*this)[i], maximization);

				switch (ComparisonResult) {
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
				std::cout << _Solution.ObjectiveValues[0] << "  " << _Solution.ObjectiveValues[1] << "  ";
				std::cout << "Exception\n";
				std::cout << "void TListSet<Solution>::Update (TPoint& Point)\n";
				std::cout << "bDominated && bDominating\n";
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
	template <class Solution> void ListSet<Solution>::GetRandomSolution(Point*& pSolution) {
		int	iIndex = 0;

		// old code
		if (iSetSize <= 0) {
			pSolution = NULL;
		}
		else {
			iIndex = rand() % iSetSize;
			pSolution = (Point*)(*this)[iIndex];
		}
	}
	template <class Solution> ListSet<Solution>::ListSet(bool maximization) {
		this->maximization = maximization;
		NumberOfObjectives = -1;
		maxBranches = 11;
		maxListSize = 20;
		nodesCalled = 0;
		nodesTested = 0;
	}
	
	template <class Solution> ListSet<Solution>::ListSet(DataSet dataset, bool maximization) {	
		NumberOfObjectives = dataset.getParameters()->NumberOfObjectives;
		maxBranches = 11;
		maxListSize = 20;
		nodesCalled = 0;
		nodesTested = 0;
		for (auto Point : dataset.points)
		{
			insert(*Point);
		}
	}
	template <class Solution> ListSet<Solution>::~ListSet() { 
		DeleteAll();
	}

	template void ListSet<Point>::UpdateNadir(int iRemovedSolution);
	template bool ListSet<Point>::insert(Point& Solution);
	template void ListSet<Point>::UpdateIdealNadir();
	template void ListSet<Point>::UpdateIdealNadir(Point& Solution);
	template void ListSet<Point>::Add(Point& Solution);
	template bool ListSet<Point>::isDominated(Point& Solution);
	template bool ListSet<Point>::checkUpdate(Point& Solution);
	template void ListSet<Point>::GetRandomSolution(Point*& pSolution);
	template ListSet<Point>::ListSet(bool maximization);
	template ListSet<Point>::ListSet(DataSet dataset, bool maximization);
	template ListSet<Point>::~ListSet();

}


#include "TreeNode.h"
namespace moda {
	template<class Solution> long TreeNode<Solution>::numberOfSolutions() {
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
	template<class Solution> long TreeNode<Solution>::numberOfNodes() {

		long sizeValue = 0;
		if (listSet.size() == 0)
			sizeValue = 1;
		unsigned i;
		for (i = 0; i < branches.size(); i++)
			sizeValue += branches[i]->numberOfNodes();
		return sizeValue;
	}
	template<class Solution> long TreeNode<Solution>::numberOfLeafs() {
		long sizeValue = 0;
		if (listSet.size() != 0)
			sizeValue = 1;
		unsigned i;
		for (i = 0; i < branches.size(); i++)
			sizeValue += branches[i]->numberOfNodes();
		return sizeValue;
	}
	//template<class Solution> bool TreeNode<Solution>::checkIdeals() {
	//	if (ComponentSolution) { // @@NEW
	//		return true;
	//	}
	//	else {
	//		bool correct = true;
	//		unsigned j;
	//		for (j = 0; j < branches.size(); j++) {
	//			ComparisonResult ComparisonResult = approximateIdealPoint.Compare(branches[j]->approximateIdealPoint);
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
	template<class Solution> void TreeNode<Solution>::print(int level, std::fstream& stream) {
		std::string s;
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
	template<class Solution> void TreeNode<Solution>::updateIdeal(TreeNode <Point>* node) {
		if (node->listSet.size() == 0 && node->branches.size() == 0)
			return;
		bool changed = false;
		unsigned j;
		for (j = 0; j < approximateIdealPoint.NumberOfObjectives; j++) {
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
	template<class Solution> void TreeNode<Solution>::updateIdealNadir() {
		unsigned i;
		for (i = 0; i < branches.size(); i++) {
			if (branches[i]->listSet.size() == 0 && branches[i]->branches.size() == 0)
				continue;
			if (branches[i]->approximateIdealPoint.ObjectiveValues[0] == 0)
				int a = 1;
			if (i == 0) {
				unsigned j;
				for (j = 0; j < MAXOBJECTIVES; j++)
				{
					approximateIdealPoint.ObjectiveValues[j] = branches[i]->approximateIdealPoint.ObjectiveValues[j];
					approximateNadirPoint.ObjectiveValues[j] = branches[i]->approximateNadirPoint.ObjectiveValues[j];
				}
			}
			else {
				unsigned j;
				for (j = 0; j < NumberOfObjectives; j++) {
					if (true) { //(Objectives[j].ObjectiveType == _Max) { //sprawdzamy, czy jest to problem maksymalizacyjny; a z za³o¿enia zawsze jest
						if (approximateIdealPoint.ObjectiveValues[j] > branches[i]->approximateIdealPoint.ObjectiveValues[j]) {
							approximateIdealPoint.ObjectiveValues[j] = branches[i]->approximateIdealPoint.ObjectiveValues[j];
						}
						if (approximateNadirPoint.ObjectiveValues[j] < branches[i]->approximateNadirPoint.ObjectiveValues[j]) {
							approximateNadirPoint.ObjectiveValues[j] = branches[i]->approximateNadirPoint.ObjectiveValues[j];
						}
					}
				}
			}
		}
		if (parent != NULL)
			parent->updateIdealNadir();

	}
	template<class Solution> void TreeNode<Solution>::updateIdealNadir(TreeNode* node)
	{
		if (node->NumberOfObjectives == -1) node->NumberOfObjectives = 2;
		if (node->listSet.size() == 0 && node->branches.size() == 0)
			return;
		bool changed = false;
		unsigned j;
		for (j = 0; j < node->NumberOfObjectives; j++) {
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
			parent->updateIdealNadir(this);
	}
	template<class Solution> DType TreeNode<Solution>::insertDistance(Point& NewSolution) {
		DType s = 0;
		int iobj; for (iobj = 0; iobj < NumberOfObjectives; iobj++) {
			//DType Range = approximateIdealPoint.ObjectiveValues[iobj] - approximateNadirPoint.ObjectiveValues[iobj];
			//if (Range == 0)
			//	Range = 1;
			//DType center = (approximateIdealPoint.ObjectiveValues[iobj] + approximateNadirPoint.ObjectiveValues[iobj]) / 2.0;
			DType center = approximateIdealPoint[iobj];
			DType s1 = (center - NewSolution.ObjectiveValues[iobj]);// / Range;
			s += s1 * s1;
		}
		return s;
	}

	
	template<class Solution> void TreeNode<Solution>::splitByClustering() {
		int numberOfClusert = std::min(maxListSize + 1, maxBranches);//8

		std::vector <Point*> seeds;

		int j;
		for (j = 0; j < numberOfClusert; j++) {
			branches.push_back(new TreeNode(this));
			if (j == 0) {
				// Find solution furthest from all other solutions
				DType maxDistance = 0;
				int maxIndex = 0;
				unsigned i;
				for (i = 0; i < listSet.size(); i++) {
					unsigned i2;
					DType sumDistance = 0;
					for (i2 = 1; i2 < listSet.size(); i2++) {
						if (i != i2)
							sumDistance += listSet[i]->Distance(*(listSet[i2]), approximateIdealPoint, approximateNadirPoint);
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
				Point* solution = (Point*)(listSet[maxIndex]);
				branches[j]->add(*solution);
				delete listSet[maxIndex];
				listSet.erase(listSet.begin() + maxIndex);
			}
			else {
				// Find solution furthest from all other clusters
				DType maxDistance;
				int maxIndex = 0;
				unsigned i;
				for (i = 0; i < listSet.size(); i++) {
					unsigned i2;
					DType sumDistance = 0;
					for (i2 = 0; i2 < seeds.size(); i2++) {
						if (i != i2)
							sumDistance += listSet[i]->Distance(*(seeds[i2]), approximateIdealPoint, approximateNadirPoint);
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
				Point* solution = (Point*)(listSet[maxIndex]);
				branches[j]->add(*solution);
				delete listSet[maxIndex];
				listSet.erase(listSet.begin() + maxIndex);
			}
		}

		// Now insert each solution to closest branch
		unsigned i;
		for (i = 0; i < listSet.size(); i++) {
			Point* solution = (Point*)listSet[i];
			DType minDistance = branches[0]->insertDistance(*solution);
			int minIndex = 0;
			for (j = 1; j < branches.size(); j++) {
				DType distanceValue = branches[j]->insertDistance(*solution);
#ifdef MFPC
				fpc++;
#endif // MFPC
				if (minDistance > distanceValue) {
					minDistance = distanceValue;
					minIndex = j;
				}
			}
			branches[minIndex]->insert(*solution);
		}
		listSet.DeleteAll();
		listSet.clear();
	}
	template<class Solution> void TreeNode<Solution>::add(Point& NewSolution)
	{
		listSet.insert(NewSolution);
		approximateIdealPoint = NewSolution;
		approximateNadirPoint = NewSolution;
		NumberOfObjectives = NewSolution.NumberOfObjectives;
	}
	template<class Solution> void TreeNode<Solution>::insert(Point& NewSolution) {
		if (listSet.size() > 0) {
			listSet.Add(NewSolution);
			if (listSet.idealUpdated || listSet.nadirUpdated) {
				approximateIdealPoint = listSet.ApproximateIdealPoint;
				approximateNadirPoint = listSet.ApproximateNadirPoint;
				if (parent != NULL) {
					parent->updateIdealNadir(this);
				}
			}
			if (listSet.size() > maxListSize) {
				splitByClustering();
			}
		}
		else {
			if (branches.size() == 0) {
				add(NewSolution);
			}
			else {
				// Find closest branch
				DType minDistance = branches[0]->insertDistance(NewSolution);
				int minIndex = 0;

				unsigned i;
				for (i = 1; i < branches.size(); i++) {
					DType distanceValue = branches[i]->insertDistance(NewSolution);
#ifdef MFPC
					fpc++;
#endif // MFPC
					if (minDistance > distanceValue) {
						minDistance = distanceValue;
						minIndex = i;
					}
				}
				branches[minIndex]->insert(NewSolution);
			}
		}
	}
	template<class Solution> void TreeNode<Solution>::update(Point& NewSolution, bool& nondominated, bool& dominated, bool& dominating, bool& equal, bool& added, bool& toInsert)
	{
		
		// Compare to approximateIdealPoint
		ComparisonResult NadirComparisonResult = NewSolution.Compare(approximateNadirPoint, this->maximization);
		if (NadirComparisonResult == _Dominated || NadirComparisonResult == _EqualSol) {
			dominated = true;
			nondominated = false;
			return;
		}
		//ComponentSolution = false; // @@NEW
		ComparisonResult IdealComparisonResult = NewSolution.Compare(approximateIdealPoint, maximization);

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
				added = listSet.checkUpdate(NewSolution);
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
					branches[i]->update(NewSolution, nondominated, dominated, dominating, equal,
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
	template<class Solution> TreeNode<Point> TreeNode<Solution>::removeDominated() {
		TreeNode<Point> newTreeNode;
		for (auto solution : listSet)
		{
			if (isNonDominatedOrEqual(*solution))
				newTreeNode.insert(*solution);
		}
		return newTreeNode;
	}
	template<class Solution> bool TreeNode<Solution>::isNonDominatedOrEqual(Point& ComparedSolution) {

		// Compare to approximateIdealPoint
		ComparisonResult NadirComparisonResult = ComparedSolution.Compare(approximateNadirPoint, maximization);

		if (NadirComparisonResult == _Dominated) {
			return false;
		}

		ComparisonResult IdealComparisonResult = ComparedSolution.Compare(approximateIdealPoint, maximization);

		if (IdealComparisonResult == _Dominating || IdealComparisonResult == _EqualSol) {
			return true;
		}
		else if (IdealComparisonResult == _Dominated || NadirComparisonResult == _Dominating || NadirComparisonResult == _EqualSol) {
			// if a leaf
			if (listSet.size() > 0) {
				return listSet.isDominated(ComparedSolution);
			}
			else {
				bool wasDominated = false;

				unsigned i;
				for (i = 0; i < branches.size() && !wasDominated; i++) {
					wasDominated = branches[i]->isDominated(ComparedSolution);
				}
				return wasDominated;
			}
		}
		// outside the box
		else {
			return false;
		}
	}
	template<class Solution> bool TreeNode<Solution>::isDominated(Point& ComparedSolution) {

		// Compare to approximateIdealPoint
		ComparisonResult NadirComparisonResult = ComparedSolution.Compare(approximateNadirPoint, maximization);

		if (NadirComparisonResult == _Dominated || NadirComparisonResult == _EqualSol) {
			return true;
		}

		ComparisonResult IdealComparisonResult = ComparedSolution.Compare(approximateIdealPoint, maximization);

		if (IdealComparisonResult == _Dominating || IdealComparisonResult == _EqualSol) {
			return false;
		}
		else if (IdealComparisonResult == _Dominated || NadirComparisonResult == _Dominating) {
			// if a leaf
			if (listSet.size() > 0) {
				return listSet.isDominated(ComparedSolution);
			}
			else {
				bool wasDominated = false;

				unsigned i;
				for (i = 0; i < branches.size() && !wasDominated; i++) {
					wasDominated = branches[i]->isDominated(ComparedSolution);
				}
				return wasDominated;
			}
		}
		// outside the box
		else {
			return false;
		}
	}
	template<class Solution> std::ostream& TreeNode<Solution>::Save(std::ostream& Stream) {
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
	template<class Solution> void TreeNode<Solution>::maxScalarizingFunction(DType& currentMax, Point& ReferencePoint, std::vector <DType>& WeightVector, BestSolution& bestSolution)
	{

		if (this->listSet.size() > 0) {
			DType MaxScalarizingFunctionValue = -11e30;
			unsigned int i;
			for (i = 0; i < this->listSet.size(); i++) {
				
				DType ScalarizingFunctionValue;
				
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
			DType MaxScalarizingFunctionValue = -1e30;
			unsigned j;
			for (j = 0; j < branches.size(); j++) {
				DType ScalarizingFunctionValue =
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

				DType ScalarizingFunctionValue = branches[j]->scalarizingFunctionValue;
				if (ScalarizingFunctionValue > currentMax) {
					branches[j]->maxScalarizingFunction(currentMax, ReferencePoint, WeightVector, bestSolution);
				}
			}
		}

	}
	template<class Solution> void TreeNode<Solution>::print(int tab)
	{
		for (int i = 0; i < tab; i++)
			std::cout << "  ";
		std::cout << branches.size() << '\n';
		for (int i = 0; i < branches.size(); i++) {
			branches[i]->print(tab + 1);
		}
	}
	template<class Solution> TreeNode<Solution>::TreeNode(bool maximization) {
		this->maximization = maximization;
		this->parent = NULL;
		approximateIdealPoint = Point(8);
		approximateNadirPoint = Point(8);
		NumberOfObjectives = -1;
		maxBranches = 11;
		maxListSize = 20;
		nodesCalled = 0;
		nodesTested = 0;
	
	}

	template<class Solution> TreeNode<Solution>::TreeNode(TreeNode* parentParam, bool maximization) {
		this->maximization = maximization;
		this->parent = parentParam;
		//NumberOfObjectives = parentParam->NumberOfObjectives;
		//approximateIdealPoint = idealPoint;
		//approximateNadirPoint = nadirPoint;
		//ComponentSolution = true;
		approximateIdealPoint = Point(8);
		approximateNadirPoint = Point(8);
		NumberOfObjectives = -1;
		maxBranches = 11;
		maxListSize = 20;
		nodesCalled = 0;
		nodesTested = 0;
	}
	template<class Solution> TreeNode<Solution>::~TreeNode() {
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




	template<class Solution> void TreeNode<Solution>::minScalarizingFunction(DType& currentMin, Point& ReferencePoint, std::vector <DType>& WeightVector, BestSolution& bestSolution) {
		if (this->listSet.size() > 0) {
			DType MinScalarizingFunctionValue = 1e30;
			unsigned int i;
			for (i = 0; i < this->listSet.size(); i++) {
				DType ScalarizingFunctionValue;

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
			DType MinScalarizingFunctionValue = 1e30;
			unsigned j;
			for (j = 0; j < branches.size(); j++) {
				DType ScalarizingFunctionValue;

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

				DType ScalarizingFunctionValue;

				ScalarizingFunctionValue = branches[j]->approximateIdealPoint.CleanChebycheffScalarizingFunctionOriginal(WeightVector, ReferencePoint);

				if (ScalarizingFunctionValue < currentMin) {
					branches[j]->minScalarizingFunction(currentMin, ReferencePoint,
						WeightVector, bestSolution);
				}
			}
		}
	}


	template long TreeNode<Point>::numberOfSolutions();
	template long TreeNode<Point>::numberOfNodes();
	template long TreeNode<Point>::numberOfLeafs();
	//template bool TreeNode<Point>::checkIdeals();
	template void TreeNode<Point>::print(int level, std::fstream& stream);
	template void TreeNode<Point>::updateIdeal(TreeNode <Point>* node);
	template void TreeNode<Point>::updateIdealNadir();
	template void TreeNode<Point>::updateIdealNadir(TreeNode* node);
	template DType TreeNode<Point>::insertDistance(Point& NewSolution);

	template void TreeNode<Point>::splitByClustering();
	template void TreeNode<Point>::print(int tab);
	template void TreeNode<Point>::add(Point& NewSolution);
	template void TreeNode<Point>::insert(Point& NewSolution);
	template void TreeNode<Point>::update(Point& NewSolution, bool& nondominated, bool& dominated, bool& dominating, bool& equal, bool& added, bool& toInsert);
	template TreeNode<Point> TreeNode<Point>::removeDominated();
	template bool TreeNode<Point>::isDominated(Point& ComparedSolution);
	template bool TreeNode<Point>::isNonDominatedOrEqual(Point& ComparedSolution);
	template std::ostream& TreeNode<Point>::Save(std::ostream& Stream);
	template void TreeNode<Point>::maxScalarizingFunction(DType& currentMax, Point& ReferencePoint, std::vector <DType>& WeightVector, BestSolution& bestSolution);
	template void TreeNode<Point>::minScalarizingFunction(DType& currentMin, Point& ReferencePoint, std::vector <DType>& WeightVector, BestSolution& bestSolution);
	template TreeNode<Point>::TreeNode(bool maximization);
	template TreeNode<Point>::TreeNode(TreeNode* parentParam, bool maximization);
	template TreeNode<Point>::~TreeNode();
}
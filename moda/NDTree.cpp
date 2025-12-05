#include "NDTree.h"
#include "DataSet.h"

namespace moda {

	std::vector<Point*> getChildren(TreeNode<Point>* node)
	{
		std::vector<Point*> listSet;
		for (auto point : node->listSet)
		{
			listSet.push_back(point);

		}
		for (auto branch : node->branches)
		{
			for (auto innerPoint : getChildren(branch))
			{
				listSet.push_back(innerPoint);
			}
		}
		return listSet;
	}
	template <class Solution> void NDTree<Solution>::saveToList() {
		unsigned i;
		//for (i = 0; i < listSet.size(); i++)
		//	delete listSet[i];
		listSet.clear();

		for (auto point : getChildren(root))
		{
			listSet.push_back(point);
		}
	}

	template <class Solution> std::string NDTree<Solution>::listToString() {
		std::stringstream sstream;
		for (auto point : listSet)
		{
			sstream << std::setprecision(2) << point->ObjectiveValues[0] << " ";
		}
		return sstream.str();
	}
	template <class Solution> long NDTree<Solution>::numberOfSolutions() {
		return root->numberOfSolutions();
	}
	template <class Solution> bool NDTree<Solution>::isDominated(Point& ComparedSolution) {
		return root->isDominated(ComparedSolution);
	}
	template <class Solution> bool NDTree<Solution>::update(Point& NewSolution, bool checkDominance) {
		
		bool nondominated = true;
		bool dominated = false;
		bool dominating = false;
		bool added = false;
		bool toInsert = false;
		bool equal = false;
		bool result = false;

		if (root == NULL) {
			root = new TreeNode <Solution>(NULL, maximization);
			root->add(NewSolution);
			result = true;
		}
		else {
			if (checkDominance) {
				root->update(NewSolution, nondominated, dominated, dominating, equal,
					added, toInsert);
			}
			if (!checkDominance || (nondominated && toInsert) || (dominating && toInsert)) {
				root->insert(NewSolution);
				result = true;
			}
			else if (nondominated) {
				root->insert(NewSolution);
				result = true;
			}
		}

		return result;
	}
	template <class Solution> bool NDTree<Solution>::update(Point& NewSolution) {
		NumberOfObjectives = NewSolution.NumberOfObjectives;
		bool nondominated = true;
		bool dominated = false;
		bool dominating = false;
		bool added = false;
		bool toInsert = false;
		bool equal = false;
		bool result = false;
		if (root == NULL) {
			root = new TreeNode<Solution>(NULL, maximization);
			root->add(NewSolution);
			result = true;
		}
		else {
			root->update(NewSolution, nondominated, dominated, dominating, equal,
				added, toInsert);
			if ((nondominated && toInsert) || (dominating && toInsert)) {
				root->insert(NewSolution);
				result = true;
			}
			else if (nondominated) {
				root->insert(NewSolution);
				result = true;
			}
		}
		return result;
	}
	template <class Solution> void NDTree<Solution>::Save(char* FileName) {
		std::fstream Stream(FileName, std::ios::out);
		root->Save(Stream);
		Stream.close();
	}
	template <class Solution> void NDTree<Solution>::DeleteAll() {
		delete root;
	}
	template <class Solution> NDTree<Solution>::NDTree(bool maximization) {
		int j;
		this->maximization = maximization;
		for (j = 0; j < MAXOBJECTIVES; j++) {
			NadirPoint.ObjectiveValues[j] = 0;
			IdealPoint.ObjectiveValues[j] = 0;
		}
	}
	template <class Solution> DataSet* NDTree<Solution>::toDataSet()
	{
		DataSet* ds = new DataSet(root->NumberOfObjectives);
		for (Point* p : root->listSet)
		{
			ds->add(p);
		}
		return ds;
	}
	template <class Solution> NDTree<Solution>::NDTree(DataSet dataset, bool maximization) {
		int j;
		this->maximization = maximization;
		NumberOfObjectives = dataset.getParameters()->NumberOfObjectives;

		Point z1 = Point::zeroes(NumberOfObjectives);
		Point z2 = Point::ones(NumberOfObjectives);

		NadirPoint = z1;
		IdealPoint = z2;


		for (auto Point : dataset.points)
		{
			update(*Point);
		}
	}
	template <class Solution> NDTree<Solution>::~NDTree() {
		DeleteAll();
	};

	template void NDTree<Point>::saveToList();
	template std::string NDTree<Point>::listToString();
	template long NDTree<Point>::numberOfSolutions();
	template bool NDTree<Point>::isDominated(Point& ComparedSolution);
	template bool NDTree<Point>::update(Point& NewSolution, bool checkDominance);
	template bool NDTree<Point>::update(Point& NewSolution);
	template void NDTree<Point>::Save(char* FileName);
	template void NDTree<Point>::DeleteAll();
	template NDTree<Point>::NDTree(bool maximization);
	template NDTree<Point>::NDTree(DataSet dataset, bool maximization);
	template NDTree<Point>::~NDTree();
	template DataSet* NDTree<Point>::toDataSet();
}
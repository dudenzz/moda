#pragma once

#include "Point.h"
#include "TreeNode.h"

namespace moda {

	class DataSet;

	template <class Solution>
	class NDTree
	{
	protected:
	public:
		std::vector <Solution*> listSet;
		Point NadirPoint;
		Point IdealPoint;
		int NumberOfObjectives;
		bool maximization;
		TreeNode<Solution>* root = NULL;
		// TODO
		void saveToList();
		std::string listToString();
		long numberOfSolutions();
		virtual bool isDominated(Point& ComparedSolution);
		virtual bool update(Point& NewSolution, bool checkDominance);
		virtual bool update(Point& NewSolution);
		
		void Save(char* FileName);
		void DeleteAll();
		DataSet* toDataSet();
		NDTree(bool maximization = true);
		NDTree(DataSet dataset, bool maximization = true);
		~NDTree();
	};


}
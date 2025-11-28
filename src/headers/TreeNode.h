#pragma once
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
#include "Point.h"
#include "ListSet.h"
#include "BestSolution.h"

namespace moda {
	template <class Solution>
	class TreeNode {
	public:
		ListSet<Solution> listSet;
	public:
		//Point approximateIdealPoint;
		//Point approximateNadirPoint;
		Point approximateIdealPoint; // TODO lowerBoundaryPoint ??
		Point approximateNadirPoint; // TODO upperBoundaryPoint ??
		std::vector <TreeNode*> branches;
		DType scalarizingFunctionValue;
		TreeNode* parent = NULL;
		//bool ComponentSolution; // @@NEW
		bool maximization;
		int NumberOfObjectives;
		int maxBranches;
		int maxListSize;
		int nodesCalled;
		int nodesTested;
		long numberOfSolutions();
		long numberOfNodes();
		long numberOfLeafs();
		// Test method only
		//bool checkIdeals();
		void print(int level, std::fstream& stream);
		void updateIdeal(TreeNode <Point>* node);
		void updateIdealNadir();
		void updateIdealNadir(TreeNode* node);
		DType insertDistance(Point& NewSolution);

		void splitByClustering();
		void add(Point& NewSolution);
		void insert(Point& NewSolution);
		virtual void update(Point& NewSolution, bool& nondominated, bool& dominated, bool& dominating, bool& equal, bool& added, bool& toInsert);
		virtual bool isDominated(Point& ComparedSolution);
		virtual bool isNonDominatedOrEqual(Point& ComparedSolution);
		TreeNode<Point> removeDominated();
		std::ostream& Save(std::ostream& Stream);
		void maxScalarizingFunction(DType& currentMax, Point& ReferencePoint, std::vector <DType>& WeightVector, BestSolution& bestSolution);
		void minScalarizingFunction(DType& currentMin, Point& ReferencePoint, std::vector <DType>& WeightVector, BestSolution& bestSolution);
		void print(int tab);
		TreeNode(bool maximization = true);
		TreeNode(TreeNode* parentParam, bool maximization = true);
		~TreeNode();
	};
}

// Copyright (C) 2015-17 Andrzej Jaszkiewicz

#pragma once
#ifndef C_LISTSET
#define C_LISTSET


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
#include "Helpers.h"
//#include "DataSet.h"

namespace moda {
	class DataSet;
	
	template <class Solution>
	class ListSet : public std::vector <Solution*>
	{
	protected:
		int	iSetSize;
		int NumberOfObjectives;
		int maxBranches;
		int maxListSize;
		int nodesCalled;
		int nodesTested;
		/// <summary>
		/// insert nadir point values
		/// Method updates nadir point approximation.
		/// Nadir point is approximated on current nondominated set.
		/// </summary>
		/// <param name="iRemovedSolution">iRemovedSolution index of solution to be removed</param>
		void UpdateNadir(int iRemovedSolution);
		
	public:
		bool idealUpdated;
		bool nadirUpdated;

		Point ApproximateIdealPoint;
		Point ApproximateNadirPoint;

		bool wasEqual;
		bool wasDominated;
		bool wasDominating;

		bool useSortedList = false;

		long updates = 0;

		long dominating = 0;
		long dominated = 0;
		long nondominated = 0;
		long equal = 0;

		bool maximization;
		/** insert set using given solution
		 *
		 * This function reduce nondominated set to given number of solutions.
		 *
		 * @param Solution possibly nondominated solution
		 * @return if true solution is nondominated and set has been updated with this solution, false solution is dominated by solution in set
		 **/
		virtual bool insert(Point& _Solution);
		void UpdateIdealNadir(Point& _Solution);
		virtual void Add(Point& _Solution);
		virtual bool isDominated(Point& _Solution);
		virtual bool checkUpdate(Point& _Solution);
		void UpdateIdealNadir();
		/** Delete all solutions from set.
		*
		* Every solution in set is released and vector is reallocated to size 0.
		**/
		virtual void DeleteAll()
		{
			//TSolutionsSet::DeleteAll();
			std::vector<Solution*>::clear();
			
			iSetSize = 0;
		}

		/// <summary>
		/// This function choose random solution from set of solutions
		/// Probability of choose for every solution should be equal.
		/// </summary>
		/// <param name="pSolution">pSolution reference to pointer where solution will be placed</param>
		virtual void GetRandomSolution(Point*& pSolution);
		/// <summary>
		/// Default non-parametrized constructor
		/// </summary>
		ListSet(bool maximization = true);
		ListSet(DataSet ds, bool maximization = true);
		~ListSet();
	};
}

#endif // ! 

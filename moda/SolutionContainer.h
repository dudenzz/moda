#include "include.h"
#include "DataSetParameters.h"
#include "Point.h"
#include "BestSolution.h"


#ifndef C_SOLUTION_CONTAINER
#define C_SOLUTION_CONTAINER
namespace moda {

	class SolutionContainer
	{
	public:
		virtual bool add(Point* newPoint) = 0;
		virtual bool remove(Point* toRemove) = 0;

		//
		//virtual DType maxScalarizingFunction(DType& currentMax, Point& ReferencePoint,
		//		vector <DType>& WeightVector, BestSolution& bestSolution) = 0;
		

		//virtual bool isDominated(Point* pointToBeChecked) = 0;
		 
		
		//virtual DataSet toDataSet() = 0;
		//virtual TreeNode<Point> toNDTree() = 0;
		//virtual ListSet<Point> toListSet() = 0;

	};
}

#endif //! C_SOLUTION_CONTAINER

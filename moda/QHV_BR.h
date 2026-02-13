#include "Solver.h"
namespace moda {
	/// <summary>
	/// This class represents code for the paper "Approximate Hypervolume calculation with guaranteed or confidence bounds" (2020)
	/// This class is dedicated for the recursive solver with upper and lower bounds estimation.
	/// This method is reffered to as QHV-BR
	/// </summary>
	
	class QHV_BR : public Solver {
	public:
		QHV_BRResult* Solve(DataSet* problem, QHV_BRParameters settings);
	private:
		DType lowerBoundVolume;
		DType upperBoundVolume;
		//recursion
		std::tuple<DType, DType, DType> initAndSolveBRHV(std::vector <Point*>& set, Point& IdealPoint, Point& NadirPoint, int numberOfSolutions, clock_t maxTime,
			std::vector <Result*>& results, bool callbacks = false);
		//recursion
		DType SolveBRHV(int start, int end, Point& IdealPoint, Point& NadirPoint, int offset,
			clock_t maxTime, clock_t& t0, clock_t& time, std::vector <Result*>& results, int recursion, int outerIteratorValue);
		DType SolveBRHV_callbacks(int start, int end, Point& IdealPoint, Point& NadirPoint, int offset,
			clock_t maxTime, clock_t& t0, clock_t& time, std::vector <Result*>& results, int recursion, int outerIteratorValue);
		inline int off(int off, int j);
		int callbackViewer = 0;
	};
}
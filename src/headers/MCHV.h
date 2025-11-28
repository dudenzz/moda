#include <tuple>
#include "Point.h"
#include "NDTree.h"
namespace moda {
	namespace backend {
		std::tuple<DType,DType,DType> solveMCHV(std::vector <Point*>& allSolutions, Point& idealPoint, Point& nadirPoint, clock_t maxTime, std::vector <MCHVResult*>& results, int numberOfObjectives);
	}
}
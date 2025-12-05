#pragma once
#include "Point.h"
#include "Hypervolume.h"
#include "SolverParameters.h"
#include "queue"
namespace moda
{
	class BHC {
	public:
		unsigned contributor;
		DType contribution;
	};


	struct compareUBHCMax {
		bool operator() (const BHC& uBHC1, const BHC& uBHC2) const {
			return uBHC1.contribution < uBHC2.contribution;
		}
	};


	struct compareUBHCMin {
		bool operator() (const BHC& uBHC1, const BHC& uBHC2) const {
			return uBHC1.contribution > uBHC2.contribution;
		}
	};
	namespace backend {
		HSSResult* greedyHSSDecLazyIQHV(std::vector <Point*>& wholeSet, std::vector <int>& selectedPoints, Point& idealPoint, Point& nadirPoint, HSSParameters::StoppingCriteriaType stopStyle, int stopSize, int stopTime, bool callbacks, bool calculateVolumeAfterEveryIteration, int numberOfObjectives);
		HSSResult* greedyHSSIncLazyIQHV(std::vector <Point*>& wholeSet, std::vector <int>& selectedPoints, Point& idealPoint, Point& nadirPoint, HSSParameters::StoppingCriteriaType stopStyle, int stopSize, int stopTime, bool callbacks, bool calculateVolumeAfterEveryIteration, int numberOfObjectives);
		DType getPointContributionIQHV(int pointIndex, std::vector <Point*>& points, Point& idealPoint, Point& nadirPoint, int numberOfObjectives);
		DType getPointContributionIQHV(Point* point, std::vector <Point*>& points, Point& idealPoint, Point& nadirPoint, int numberOfObjectives);
		DType solveIQHV(std::vector<Point*>& points, Point& idealPoint, Point& nadirPoint, int numberOfObjectives, int numberOfPoints = -1);
		

	}
}
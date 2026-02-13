#include "MCHV.h"
#include "random"
#include "Hypervolume.h"
/*************************************************************************

Quick Extreme Hypervolume Contributor

 ---------------------------------------------------------------------

						   Copyright (C) 2025
		  Andrzej Jaszkiewicz <ajaszkiewicz@cs.put.poznan.pl>
		  Piotr Zielniewicz <piotr.zielniewicz@cs.put.poznan.pl>
		  Jakub Dutkiewicz <jakub.dutkiewicz@put.poznan.pl>

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at https://mozilla.org/MPL/2.0/.

 ----------------------------------------------------------------------

 Relevant literature:

 [1] A. Jaszkiewicz and P. Zielniewicz, 
   "Improving the Efficiency of the Distance-Based Hypervolume Estimation Using ND-Tree," 
   in IEEE Transactions on Evolutionary Computation, vol. 29, no. 3, pp. 726-733, June 2025, doi: 10.1109/TEVC.2024.3391857. 

*************************************************************************/

namespace moda {
	namespace backend {
		std::tuple<DType, DType, DType>  solveMCHV(std::vector <Point*>& allSolutions, Point& idealPoint, Point& nadirPoint, clock_t maxTime, std::vector <MCHVResult*>& results, int numberOfObjectives) {

			DType z = 2.5758; // https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval https://planetcalc.com/4987/
			std::uniform_real_distribution<DType> unif(0, 1);
			std::default_random_engine re;
			DType a_random_DType = unif(re);

			NDTree<Point> NDTree;
			NDTree.NumberOfObjectives = numberOfObjectives;
			for (auto s : allSolutions) {
				NDTree.update(*s, true);
			}

			DType approximateVolume = 0;
			int dominated = 0;
			int tested = 0;

			clock_t t0 = clock();
			clock_t time = t0;

			Point randomPoint = Point(numberOfObjectives);
			while (clock() - t0 < maxTime) {
				int j;
				for (j = 0; j < numberOfObjectives; j++) {
					randomPoint.ObjectiveValues[j] = unif(re) * (idealPoint[j] - nadirPoint[j]) + nadirPoint[j];
				}
				if (NDTree.isDominated(randomPoint)) {
					dominated++;
				}
				else {
					NDTree.isDominated(randomPoint);
				}
				tested++;


				time = clock();

				MCHVResult* result = new MCHVResult;
				result->ElapsedTime = time - t0;
				DType scaler = Hypervolume(&nadirPoint, &idealPoint, numberOfObjectives);
				approximateVolume = scaler * (DType)dominated / tested;
				DType np = dominated;
				DType n = tested;
				DType pe = approximateVolume;
				DType sq = z * sqrt(z * z - 1 / n + 4 * n * pe * (1 - pe) + (4 * pe - 2)) + 1;
				DType denominator = 2 * (n + z * z);

				result->HyperVolumeEstimation = approximateVolume;
#if DTypeN == 1
				result->LowerBound = std::max(0.0f, (2 * n * pe + z * z - sq) / denominator);
				result->UpperBound = std::min(1.0f, (2 * n * pe + z * z + sq) / denominator);
#elif DTypeN == 2
				result->LowerBound = std::max(0.0, (2 * n * pe + z * z - sq) / denominator);
				result->UpperBound = std::min(1.0, (2 * n * pe + z * z + sq) / denominator);
#endif	

				result->type = Result::ResultType::Estimation;
				//IterationCallback(callbackViewer, 0, result);
				//callbackViewer += 1;
				results.push_back(result);

			}

			return std::tuple<DType, DType, DType>(results[results.size() - 1]->HyperVolumeEstimation, results[results.size() - 1]->LowerBound, results[results.size() - 1]->UpperBound);
		}
	}
}
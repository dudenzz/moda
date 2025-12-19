#include "HSS.h"
#include "IQHV.h"
#include "MemoryManager.h"
#include "ExecutionContext.h"
#include "ExecutionPool.h"
#include "ExecutionService.h"
namespace moda
{
	namespace backend {

		DType getPointContributionIQHV(Point* point, std::vector <Point*>& points, Point& idealPoint, Point& nadirPoint, int numberOfObjectives) {
			return getPointContributionIQHV(std::find(points.begin(), points.end(), point) - points.begin(), points, idealPoint, nadirPoint, numberOfObjectives);
		}
		DType getPointContributionIQHV(int pointIndex, std::vector <Point*>& points, Point& idealPoint, Point& nadirPoint, int numberOfObjectives) {
			std::vector <Point*> tmpPoints = points;
			tmpPoints.erase(tmpPoints.begin() + pointIndex);

			Point newIdealPoint;
			for (short j = 0; j < numberOfObjectives; j++) {
				newIdealPoint.ObjectiveValues[j] = points[pointIndex]->ObjectiveValues[j];
			}

			DType largeHV = Hypervolume(&nadirPoint, &newIdealPoint, numberOfObjectives);
			DType complement = solveIQHV(tmpPoints, newIdealPoint, nadirPoint, numberOfObjectives);
			return largeHV - complement;
		}


		DType solveIQHV(std::vector <Point*>& points, Point& idealPoint, Point& nadirPoint, int numberOfObjectives, int numberOfPoints) {
			// allocate memory


			if (numberOfPoints == -1) {
				numberOfPoints = points.size();
			}
			int reserve_size = 4 * numberOfPoints * pow(2, numberOfObjectives / 2);
			ExecutionService* service = &(ExecutionService::getInstance());
			ExecutionPool* pool = &(service->getPool());
			int memoryKey = pool->reserveContext(reserve_size,numberOfPoints,numberOfObjectives,ExecutionContext::ExecutionContextType::IQHVContext);
			auto context = (IQHVExecutionContext*)pool->getContext(memoryKey);
			for (int i = 0; i < numberOfPoints; i++) {
				(*context->points)[i] = new Point(*points[i]);
			}


			//DType result = IQHV_for_hss(0, numberOfPoints - 1, idealPoint, nadirPoint, 0, numberOfObjectives, memoryKey, maxIndexKey);
			int maxIndexMem = numberOfPoints - 1;
			context->maxIndexUsed = maxIndexMem;
			DType result = IQHV(0, numberOfPoints - 1, memoryKey, idealPoint, nadirPoint, 0, numberOfObjectives, 0,  numberOfPoints);
			pool->releaseContext(memoryKey);
			// release memory
			//indexSetVec.clear();
			//indexSetVec.shrink_to_fit();
			return result;
		}
		
		HSSResult* greedyHSSIncLazyIQHV(std::vector <Point*>& wholeSet, std::vector <int>& selectedPoints, Point& idealPoint, Point& nadirPoint, HSSParameters::StoppingCriteriaType stopStyle, int stopSize, int stopTime, bool callbacks, bool calculateVolumeAfterEveryIteration, int numberOfObjectives) {

			if (stopStyle == HSSParameters::StoppingCriteriaType::Time) stopSize = 0;
			if (stopStyle == HSSParameters::StoppingCriteriaType::SubsetSize) stopTime = INT32_MAX;
			HSSResult* result = new HSSResult();
			clock_t t0 = clock();
			std::vector <Point*> subset;
			std::priority_queue<BHC, std::vector<BHC>, compareUBHCMax> queueUBHC;
			DType contribution;
			unsigned numberOfPoints = wholeSet.size();
			bool initialLoop = true;
			DType totalContribution = 0;
			std::vector <bool> selected;
			selected.resize(wholeSet.size(), false);

			while (subset.size() < stopSize) {
				if (clock() - t0 > stopTime && stopStyle == 2) break;
				std::vector<BHC> newUBHC;
				DType maxContribution = -1e30;
				int maxContributor = -1;

				if (initialLoop) {
					for (unsigned j = 0; j < wholeSet.size(); j++) {
						if (!selected[j]) {
							subset.push_back(wholeSet[j]);

							contribution = getPointContributionIQHV(subset.size() - 1, subset, idealPoint, nadirPoint, numberOfObjectives);
							subset.pop_back();

							if (maxContribution < contribution) {
								maxContribution = contribution;
								maxContributor = j;
							}

							BHC uBHC;
							uBHC.contributor = j;
							uBHC.contribution = contribution;
							queueUBHC.push(uBHC);
						}
					}
					initialLoop = false;
				}
				else {
					DType prevContribution = 1;

					while (queueUBHC.size() > 0) {
						BHC uBHC = queueUBHC.top();
						prevContribution = uBHC.contribution;

						if (uBHC.contribution < maxContribution) {
							break;
						}

						queueUBHC.pop();
						subset.push_back(wholeSet[uBHC.contributor]);
						
						contribution = getPointContributionIQHV(subset.size() - 1, subset, idealPoint, nadirPoint, numberOfObjectives);
						subset.pop_back();

						if (maxContribution < contribution) {
							maxContribution = contribution;
							maxContributor = uBHC.contributor;
						}

						BHC uBHCTemp;
						uBHCTemp.contributor = uBHC.contributor;
						uBHCTemp.contribution = contribution;
						newUBHC.push_back(uBHCTemp);
					}

					for (auto uBHC : newUBHC) {
						if (uBHC.contributor != maxContributor) {
							queueUBHC.push(uBHC);
						}
					}
				}

				//cout << maxContributor << ' ';
				subset.push_back(wholeSet[maxContributor]);
				selected[maxContributor] = true;
				selectedPoints.push_back(maxContributor);
				if (calculateVolumeAfterEveryIteration)
				{
					totalContribution += maxContribution;
				}
				//if (callbacks)
				//{
				//	result->type = Result::SubsetSelection;
				//	result->selectedPoints = selectedPoints;
				//	result->HyperVolume = totalContribution;
				//	result->chosenPointIndex = maxContributor;
				//	IterationCallback(subset.size(), stopSize, result);
				//}

			}

			result->type = Result::SubsetSelection;
			result->selectedPoints = selectedPoints;
			//result->HyperVolume = solveIQHV(subset, idealPoint, nadirPoint);
			return result;


		}

		HSSResult* greedyHSSDecLazyIQHV(std::vector <Point*>& wholeSet, std::vector <int>& selectedPoints, Point& idealPoint, Point& nadirPoint, HSSParameters::StoppingCriteriaType stopStyle, int stopSize, int stopTime, bool callbacks, bool calculateVolumeAfterEveryIteration, int numberOfObjectives) {
			clock_t t0 = clock();

			if (stopStyle == HSSParameters::StoppingCriteriaType::Time) stopSize = 0;
			if (stopStyle == HSSParameters::StoppingCriteriaType::SubsetSize) stopTime = INT32_MAX;
			HSSResult* result = new HSSResult();
			std::priority_queue<BHC, std::vector<BHC>, compareUBHCMin> queueUBHC;
			DType contribution;

			std::vector <Point*> subset = wholeSet;
			bool initialLoop = true;

			unsigned j;
			for (j = 0; j < wholeSet.size(); j++) {
				selectedPoints.push_back(j);
			}


			int initialSize = subset.size();

			DType totalVolume = 0;
			if (calculateVolumeAfterEveryIteration) {
				//totalVolume = solveIQHV(subset, idealPoint, nadirPoint);
			}

			while (subset.size() > stopSize) {
				if (clock() - t0 > stopTime && stopStyle == 2) break;
				std::vector<BHC> newLBHC;
				DType minContribution = 1e30;
				int minContributor = -1;

				if (initialLoop) {
					for (j = 0; j < wholeSet.size(); j++) {
						contribution = getPointContributionIQHV(j, subset, idealPoint, nadirPoint, numberOfObjectives);

						if (minContribution > contribution) {
							minContribution = contribution;
							minContributor = j;
						}

						BHC lBHC;
						lBHC.contributor = j;
						lBHC.contribution = contribution;
						//IterationCallback(j, wholeSet.size(), result);
						queueUBHC.push(lBHC);
					}
					queueUBHC.pop();
					initialLoop = false;
				}
				else {
					DType prevContribution = 1;

					while (queueUBHC.size() > 0) {
						BHC lBHC = queueUBHC.top();
						prevContribution = lBHC.contribution;

						if (lBHC.contribution > minContribution) {
							break;
						}

						queueUBHC.pop();
						contribution = getPointContributionIQHV(wholeSet[lBHC.contributor], subset, idealPoint, nadirPoint, numberOfObjectives);

						if (minContribution > contribution) {
							minContribution = contribution;
							minContributor = lBHC.contributor;
						}

						BHC uBHCTemp;
						uBHCTemp.contributor = lBHC.contributor;
						uBHCTemp.contribution = contribution;

						newLBHC.push_back(uBHCTemp);
					}

					for (auto lBHC : newLBHC) {
						if (lBHC.contributor != minContributor) {
							queueUBHC.push(lBHC);
						}
					}
				}
				//debug
			/*	DetailedSubsetSelectionResult* detailed_result = new DetailedSubsetSelectionResult();
				detailed_result->type = Result::DetailedSubsetSelection;
				vector<PointContribution*> v;
				for (Point* point : subset)
				{
					PointContribution* pc = new PointContribution();
					pc->index = find(wholeSet.begin(), wholeSet.end(), point) - wholeSet.begin();
					pc->value = getPointContributionIQHV(point, subset, idealPoint, nadirPoint);
					v.push_back(pc);
				}
				detailed_result->selectedPoints = v;
			*/
			//!!debug
				int minContributorIndex = find(subset.begin(), subset.end(), wholeSet[minContributor]) - subset.begin();
				//detailed_result->ereased_index = minContributor;
				subset.erase(subset.begin() + minContributorIndex);
				selectedPoints.erase(selectedPoints.begin() + minContributorIndex);
				if (calculateVolumeAfterEveryIteration)
				{
					totalVolume -= minContribution;
				}
				if (callbacks)
				{
					result->type = Result::SubsetSelection;
					result->selectedPoints = selectedPoints;
					result->HyperVolume = totalVolume;
					result->chosenPointIndex = minContributor;
					//IterationCallback(initialSize - subset.size(), initialSize - stopSize, result);
				}
			}

			result->type = Result::SubsetSelection;
			result->HyperVolume = solveIQHV(subset, idealPoint, nadirPoint, numberOfObjectives);
			result->selectedPoints = selectedPoints;
			//std::cout << subset.size();
			return result;
		}
	}
}
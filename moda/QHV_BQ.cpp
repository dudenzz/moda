#pragma once
#include "QHV_BQ.h"
#include "Hypervolume.h"
#include "SPData.h"
#include "DynamicStructures.h"
#include "ExecutionContext.h"
#include "ExecutionPool.h"
#include "ExecutionService.h"
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

 [1] Andrzej Jaszkiewicz and Piotr Zielniewicz. 2021. 
     Quick extreme hypervolume contribution algorithm.
     In Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '21). 
     Association for Computing Machinery, New York, NY, USA, 412–420. 
     https://doi.org/10.1145/3449639.3459394

*************************************************************************/
namespace moda {
	namespace backend {

		std::tuple<DType, DType, DType>  QHV_BQ(std::vector <Point*>& set, Point& idealPoint, Point& nadirPoint, int numberOfSolutions, clock_t maxTime, bool switch2MC,
			std::vector < QHV_BQResult* >& results, SwitchParameters mcSettings, int numberOfObjectives)
		{

			myvector<Point*> indexSet;
			indexSet.resize(10000000);
			int branches = 0;

			Point* point2 = new Point;
			unsigned int i;
			for (i = 0; i < numberOfSolutions; i++) {
				if ((set)[i] == NULL) {
					continue;
				}
				indexSet[i] = set[i];
			}

			Point newIdealPoint = Point(numberOfObjectives), newNadirPoint = Point(numberOfObjectives);
			int j;

			//THIS CHANGE IS DESIGNED FOR DEFINEABLE NADIR AND IDEAL POINTS (NOT A HARDCODED ONES AND ZEROES)
			//for (j = 0; j < currentSettings.NumberOfObjectives; j++) {
			//	newIdealPoint.ObjectiveValues[j] = 1;
			//	newNadirPoint.ObjectiveValues[j] = 0;
			//}
			newIdealPoint = idealPoint;
			newNadirPoint = nadirPoint;

			int maxIndexUsed = numberOfSolutions - 1;

			SubProblem* newSubproblem = new SubProblem;
			newSubproblem->IdealPoint = newIdealPoint;
			newSubproblem->NadirPoint = newNadirPoint;
			newSubproblem->start = 0;
			newSubproblem->end = numberOfSolutions - 1;
			newSubproblem->volume = Hypervolume(&newSubproblem->NadirPoint, &newSubproblem->IdealPoint, numberOfObjectives);

			DType upperBoundVolume = newSubproblem->volume;
			DType lowerBoundVolume = 0;

			clock_t t0 = clock();
			clock_t time = t0;

			//	SubProblemsStackPriorityQueue subProblemsStack;
			SubproblemsPool<SubProblem>* subProblems = new SubproblemsPool<SubProblem>();
			SubProblemsStackLevel subProblemsStack(10);

			subProblemsStack.subProblems = subProblems;
			int iSP = subProblems->getNew();
			(*subProblems)[iSP].end = newSubproblem->end;
			(*subProblems)[iSP].start = newSubproblem->start;
			(*subProblems)[iSP].NadirPoint = newSubproblem->NadirPoint;
			(*subProblems)[iSP].IdealPoint = newSubproblem->IdealPoint;
			(*subProblems)[iSP].volume = newSubproblem->volume;

			subProblemsStack.push_back(iSP);

			int offset = 0;
			Point randomPoint;

			clock_t tm0 = clock();

			bool run = true;
			while (subProblemsStack.size() > 0 && run) {
				if (branches % 1000 == 0) {
					run = clock() - t0 < maxTime;
					int sps = subProblemsStack.size();
					run = run && (!switch2MC || subProblemsStack.size() <= mcSettings.maxStackProblemSize);
					run = run && (!switch2MC || clock() - t0 < mcSettings.switchTime);
					run = run && (!switch2MC || mcSettings.gap < upperBoundVolume - lowerBoundVolume);
					//QHV_BQResult* result = new QHV_BQResult();
					//result->ElapsedTime = time - t0;
					//result->HyperVolumeEstimation = (lowerBoundVolume + upperBoundVolume) / 2;
					//result->LowerBound = lowerBoundVolume;
					//result->UpperBound = upperBoundVolume;
					//result->type = Result::Estimation;
					//results.push_back(result);
					//IterationCallback(callbackViewer, 0, result);
				}

				branches++;
				int iSP = subProblemsStack.back();
				{
					subProblemsStack.pop_back();
				}

				const SubProblem* pSP = &((*subProblems).get(iSP));

				offset = (++offset) % numberOfObjectives;

				int oldmaxIndexUsed = maxIndexUsed;

				// If there is just one point
				if (pSP->start == pSP->end) {
					DType v = backend::Hypervolume(&(pSP->NadirPoint), indexSet[pSP->start], &(pSP->IdealPoint), numberOfObjectives);
					lowerBoundVolume += v;
					upperBoundVolume -= pSP->volume - v;
					(*subProblems).free(iSP);
					continue;
				}

				// If there are just two points
				if (pSP->end - pSP->start == 1) {
					DType v = backend::Hypervolume(&pSP->NadirPoint, indexSet[pSP->start], &pSP->IdealPoint, numberOfObjectives);
					v += backend::Hypervolume(&pSP->NadirPoint, indexSet[pSP->end], & pSP->IdealPoint, numberOfObjectives);
					*point2 = *(indexSet[pSP->start]);
					unsigned j;
					for (j = 0; j < numberOfObjectives; j++) {
						point2->ObjectiveValues[j] = std::min(point2->ObjectiveValues[j], indexSet[pSP->end]->ObjectiveValues[j]);
					}
					v -= backend::Hypervolume(&pSP->NadirPoint, point2, &pSP->IdealPoint, numberOfObjectives);

					lowerBoundVolume += v;
					upperBoundVolume -= pSP->volume - v;

					(*subProblems).free(iSP);
					continue;
				}

				upperBoundVolume -= pSP->volume;

				int iPivot = pSP->start;
				DType maxVolume;
				Point p = *(indexSet)[iPivot];
				maxVolume = backend::Hypervolume(&pSP->NadirPoint, &p, &pSP->IdealPoint, numberOfObjectives);

				// Find the pivot point
				unsigned i;
				for (i = pSP->start + 1; i <= pSP->end; i++) {
					DType volumeCurrent;
					volumeCurrent = backend::Hypervolume(&pSP->NadirPoint, indexSet[i], &pSP->IdealPoint, numberOfObjectives);
					if (maxVolume < volumeCurrent) {
						maxVolume = volumeCurrent;
						iPivot = i;
					}
				}

				DType v = backend::Hypervolume(&pSP->NadirPoint, indexSet[iPivot], &pSP->IdealPoint, numberOfObjectives);
				lowerBoundVolume += v;
				upperBoundVolume += v;

				unsigned j;
				int jj;

				Point partNadirPoint = Point(pSP->NadirPoint.NumberOfObjectives);
				for (int i = 0; i < pSP->NadirPoint.NumberOfObjectives; i++)
					partNadirPoint[i] = pSP->NadirPoint[i];

				Point partIdealPoint = Point(pSP->IdealPoint.NumberOfObjectives);
				for (int i = 0; i < pSP->IdealPoint.NumberOfObjectives; i++)
					partIdealPoint[i] = pSP->IdealPoint[i];


				int iPos = maxIndexUsed + 1;
				Point* pPivotPoint = (indexSet)[iPivot];

				// Recursive call for each subproblem
				for (jj = 0; jj < numberOfObjectives; jj++) {
					j = off(offset, jj,numberOfObjectives);

					if (jj > 0) {
						int j2 = off(offset, jj - 1,numberOfObjectives);
						partIdealPoint.ObjectiveValues[j2] = std::min(pSP->IdealPoint.ObjectiveValues[j2], pPivotPoint->ObjectiveValues[j2]);
						partNadirPoint.ObjectiveValues[j2] = pSP->NadirPoint.ObjectiveValues[j2];
					}

					int partStart = iPos;
					for (i = pSP->start; i <= pSP->end; i++) {

						if (i == iPivot)
							continue;

						if (std::min(pSP->IdealPoint.ObjectiveValues[j], indexSet[i]->ObjectiveValues[j]) >
							std::min(pSP->IdealPoint.ObjectiveValues[j], pPivotPoint->ObjectiveValues[j])) {
							indexSet[iPos++] = indexSet[i];
						}
					}

					int partEnd = iPos - 1;
					maxIndexUsed = iPos - 1;

					if (partEnd >= partStart) {
						partNadirPoint.ObjectiveValues[j] = std::min(pSP->IdealPoint.ObjectiveValues[j], pPivotPoint->ObjectiveValues[j]);

						DType v = Hypervolume(&partNadirPoint, &partIdealPoint,numberOfObjectives);
						upperBoundVolume += v;

						int newISP = (*subProblems).getNew();
						SubProblem* pNewSP = &((*subProblems)[newISP]);

						pNewSP->start = partStart;
						pNewSP->end = partEnd;
						pNewSP->volume = v;
						pNewSP->IdealPoint = partIdealPoint;
						pNewSP->NadirPoint = partNadirPoint;
						subProblemsStack.push_back(newISP);
					}

				}

				(*subProblems).free(iSP);

				if (maxIndexUsed > indexSet.size() - 10000000) {
					indexSet.resize(indexSet.size() + 10000000);
				}
			}

			if (switch2MC) {
				return solveMC(set, subProblemsStack, maxTime, t0, time, results,subProblems,upperBoundVolume,lowerBoundVolume,numberOfObjectives);
			}

			delete newSubproblem;
			indexSet.clear();
			(*subProblems).clear();

			return std::tuple<DType, DType, DType>(lowerBoundVolume, (lowerBoundVolume + upperBoundVolume) / 2.0, upperBoundVolume);
		}
		std::tuple<DType, DType, DType>  solveMC(std::vector <Point*>& allSolutions, SubProblemsStackLevel& subProblemsStack, clock_t maxTime, clock_t t0, clock_t time,
			std::vector <QHV_BQResult*>& results, SubproblemsPool<SubProblem>* subProblems, DType upperBoundVolume, DType lowerBoundVolume, int numberOfObjectives) {

			

			SPData sPData;
			std::vector <SPData> remainingSubProblems;
			subProblemsStack.startIterating();

			DType sumVolume = 0;
			int iSP = subProblemsStack.next();
			int iter = 0;

			while (iSP >= 0) {
				iter++;
				sPData.id = iSP;
				sumVolume += (*subProblems)[iSP].volume;
				sPData.sumVolume = sumVolume;
				remainingSubProblems.push_back(sPData);
				iSP = subProblemsStack.next();
			}

			DType z = 2.5758; // https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval https://planetcalc.com/4987/
			std::uniform_real_distribution<DType> unifDType(0, 1);
			std::default_random_engine re;

			// Put all points in ND-Tree
			NDTree<Point> NDTree;

			for (Point* s : allSolutions) {
				NDTree.update(*s);
			}

			DType approximateVolume = 0;
			int dominated = 0;
			int tested = 0;

			DType unknownVolume = upperBoundVolume - lowerBoundVolume;

			Point randomPoint = Point(numberOfObjectives);
			while (clock() - t0 < maxTime) {
				DType rv = unifDType(re) * unknownVolume;

				int randomSP;
				int iSPMin = 0;
				int iSPMax = remainingSubProblems.size() - 1;
				while (iSPMax - iSPMin > 1) {
					int iSPMid = (iSPMin + iSPMax) / 2;
					if (rv <= remainingSubProblems[iSPMid].sumVolume) {
						iSPMax = iSPMid;
					}
					else {
						iSPMin = iSPMid;
					}
				}
				randomSP = remainingSubProblems[iSPMax].id;

				DType volumeFraction = (*subProblems)[randomSP].volume / unknownVolume;

				int j;
				for (j = 0; j < numberOfObjectives; j++) {
					randomPoint.ObjectiveValues[j] = (*subProblems)[randomSP].NadirPoint.ObjectiveValues[j] + unifDType(re) *
						((*subProblems)[randomSP].IdealPoint.ObjectiveValues[j] - (*subProblems)[randomSP].NadirPoint.ObjectiveValues[j]);
				}

				if (NDTree.isDominated(randomPoint)) {
					dominated++;
				}
				else {
					dominated = dominated;
					NDTree.isDominated(randomPoint);
				}
				tested++;


				time = clock();

//				QHV_BQResult* result = new QHV_BQResult();
//				result->ElapsedTime = time - t0;
//
//				DType np = dominated;
//				DType n = tested;
//				DType pe = np / n;
//				DType sq = z * sqrt(z * z - 1 / n + 4 * n * pe * (1 - pe) + (4 * pe - 2)) + 1;
//				DType denominator = 2 * (n + z * z);
//
//				result->HyperVolumeEstimation = np / n * unknownVolume + lowerBoundVolume;
//#if DTypeN == 1
//				result->LowerBound = std::max(0.0f, (2 * n * pe + z * z - sq) / denominator);
//				result->UpperBound = std::min(1.0f, (2 * n * pe + z * z + sq) / denominator);
//#elif DTypeN == 2
//				result->LowerBound = std::max(0.0, (2 * n * pe + z * z - sq) / denominator);
//				result->UpperBound = std::min(1.0, (2 * n * pe + z * z + sq) / denominator);
//#endif	
//
//				result->type = Result::ResultType::Estimation;
//				results.push_back(result);
//				IterationCallback(callbackViewer, 0, result);
//				callbackViewer += 1;

			}

			return std::tuple<DType, DType, DType>(results[results.size() - 1]->HyperVolumeEstimation, results[results.size() - 1]->LowerBound, results[results.size() - 1]->UpperBound);

		}


	}
}

#include "QEHC.h"
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
     Association for Computing Machinery, New York, NY, USA, 412�420. 
     https://doi.org/10.1145/3449639.3459394

*************************************************************************/
namespace moda {
	namespace backend {
		bool sortByPointsCounterAsc(SubproblemParam lhs, SubproblemParam rhs) {
			return lhs.pointsCounter < rhs.pointsCounter;
		}
		QEHCResult QEHC(int contextId, int numberOfSolutions, int maxlevel, QEHCParameters::SearchSubjectOption searchSubject, bool useSort, bool useShuffle, int offset, unsigned long int iterationLimit, int numberOfObjectives)
		{
			QEHCResult tmpResult;
			std::random_device randomDevice;
			std::mt19937 randomNumberGenerator(randomDevice());
			SubproblemsPool <SubProblem> subProblems;
			#if UNDERLYING_TYPE == 1
				myvector<Point*>* indexSet;
			#elif UNDERLYING_TYPE == 2
				SemiDynamicArray<Point*>* indexSet;			
			#elif UNDERLYING_TYPE == 3
				SecureVector<Point*>* indexSet;
			#else
						std::vector<Point*>* indexSet;
			#endif

			backend::ExecutionService* poolService = &(backend::ExecutionService::getInstance());
			backend::ExecutionPool* pool = &(poolService->getPool());
						
			backend::QEHCExecutionContext* context = (backend::QEHCExecutionContext*)pool->getContext(contextId);

			indexSet = context->points;
			int maxIndexUsed = context->maxIndexUsed;
			//if (!useDelete) {
			//	iterationLimit = ULONG_MAX;
			//}
			bool toDelete;
			clock_t t0 = clock();


			Point* point2 = new Point;

			Point newNadirPoint;
			int j;
			for (j = 0; j < numberOfObjectives; j++) {
				newNadirPoint.ObjectiveValues[j] = 0;
			}

			DType maxContributionLowerBound = 0;
			DType minContributionUpperBound = 1;
			int lowerBoundProcessId = -1;
			int upperBoundProcessId = -1;

			int pos = context->initialSize;
			unsigned long int iterLimit = iterationLimit;
			std::vector <ProcessData*> processes;
			processes.resize(numberOfSolutions);

			int ip;
			int ip0;
			unsigned long int ii;
			for (ip = 0; ip < numberOfSolutions; ip++) {
				processes[ip] = new ProcessData(maxlevel);
				int iSP = subProblems.getNew();
				subProblems[iSP].IdealPoint = *((*indexSet)[ip]);
				subProblems[iSP].NadirPoint = newNadirPoint;
				subProblems[iSP].start = pos;
				subProblems[iSP].end = pos + numberOfSolutions - 2;
				subProblems[iSP].volume = Hypervolume(&subProblems[iSP].NadirPoint, &subProblems[iSP].IdealPoint, numberOfObjectives);
				processes[ip]->lowerBoundVolume = 0;
				processes[ip]->upperBoundVolume = subProblems[iSP].volume;
				processes[ip]->totalVolume = subProblems[iSP].volume;
				processes[ip]->subProblemsStack.subProblems = &subProblems;
				processes[ip]->subProblemsStack.push_back(iSP);
				processes[ip]->id = ip;

				int i3 = 0;
				int i2; for (i2 = 0; i2 < numberOfSolutions; i2++) {
					if (i2 != ip) {
						//if ((set)[i2] == NULL) {
						//	continue;
						//}
						(*indexSet)[pos + i3++] = (*indexSet)[i2];
					}
				}
				pos += numberOfSolutions - 1;
			}
			maxIndexUsed = pos - 1;
			int oldmaxIndexUsed = maxIndexUsed;

			//if (processId > 0) {
			//	swap(processes[0], processes[processId]);
			//	iterLimit = ULONG_MAX;
			//}
			//else
			//{
			//	iterLimit = iterationLimit;
			//}

			// use line below together with rotate function (instead of sort function on subproblemParams)
			//int offset = 0;

			long nIterations = 0;
			bool runningPhase1 = true;
			while (processes.size() > 0) {

				for (ip = 0; ip < processes.size(); ip++) {

					if (runningPhase1) {
						//if (processId < 0) {
						if (processes.size() == 1) {

							runningPhase1 = false;
							//cout << "  Phase #1 - process id:\t" << processes[0]->id << "\n";
							//cout << "  Phase #1 - iterations:\t" << nIterations << "\n";
						}
						//}
						//else if (processes[ip]->id != processId) {
						//	
						//	runningPhase1 = false;
						//}
					}

					//if (!useDelete && processes.size() % 50 == 0) {
					//	cout << "#";
					//}

					for (ii = 0; ii < iterLimit; ii++) {
						//std::cout << minContributionUpperBound << "\n";
						if (processes[ip]->subProblemsStack.size() == 0) {
							delete processes[ip];
							processes.erase(processes.begin() + ip);
							ip--;
							break;
						}

						nIterations++;
						int iSP = processes[ip]->subProblemsStack.back();
						{
							processes[ip]->subProblemsStack.pop_back();
						}
						

						if (!useShuffle && !useSort)
						{
							offset++;
							offset = offset % numberOfObjectives;
						}

						// If there is just one point
						if (subProblems[iSP].start == subProblems[iSP].end) {
							DType v = backend::Hypervolume(&subProblems[iSP].NadirPoint, ((*indexSet)[subProblems[iSP].start]), &subProblems[iSP].IdealPoint, numberOfObjectives);
							processes[ip]->lowerBoundVolume += v;
							processes[ip]->upperBoundVolume -= subProblems[iSP].volume - v;
							subProblems.free(iSP);

							if (maxContributionLowerBound < processes[ip]->totalVolume - processes[ip]->upperBoundVolume) {
								maxContributionLowerBound = processes[ip]->totalVolume - processes[ip]->upperBoundVolume;
								lowerBoundProcessId = processes[ip]->id;
							}
							if (minContributionUpperBound > processes[ip]->totalVolume - processes[ip]->lowerBoundVolume) {
								minContributionUpperBound = processes[ip]->totalVolume - processes[ip]->lowerBoundVolume;
								upperBoundProcessId = processes[ip]->id;
							}

							//if (useDelete) {
							toDelete = false;
							if (searchSubject == QEHCParameters::SearchSubjectOption::MinimumContribution) {
								if (processes[ip]->totalVolume - processes[ip]->upperBoundVolume > minContributionUpperBound) {
									toDelete = true;
								}
							}
							else {
								if (processes[ip]->totalVolume - processes[ip]->lowerBoundVolume < maxContributionLowerBound) {
									toDelete = true;
								}
							}
							if (toDelete) {
								delete processes[ip];
								processes.erase(processes.begin() + ip);
								ip--;
								break;
							}
							//}
							continue;
						}

						// If there are just two points
						if (subProblems[iSP].end - subProblems[iSP].start == 1) {
							DType v = backend::Hypervolume(&subProblems[iSP].NadirPoint, ((*indexSet)[subProblems[iSP].start]), &subProblems[iSP].IdealPoint, numberOfObjectives);
							v += backend::Hypervolume(&subProblems[iSP].NadirPoint, ((*indexSet)[subProblems[iSP].end]), &subProblems[iSP].IdealPoint, numberOfObjectives);
							*point2 = *((*indexSet)[subProblems[iSP].start]);
							Point p3 = *((*indexSet)[subProblems[iSP].end]);
							unsigned j;
							for (j = 0; j < numberOfObjectives; j++) {
								
								point2->ObjectiveValues[j] = std::min(point2->ObjectiveValues[j], (*indexSet)[subProblems[iSP].end]->ObjectiveValues[j]);
							}
							v -= backend::Hypervolume(&subProblems[iSP].NadirPoint, point2, &subProblems[iSP].IdealPoint, numberOfObjectives);

							processes[ip]->lowerBoundVolume += v;
							processes[ip]->upperBoundVolume -= subProblems[iSP].volume - v;
							subProblems.free(iSP);

							if (maxContributionLowerBound < processes[ip]->totalVolume - processes[ip]->upperBoundVolume) {
								maxContributionLowerBound = processes[ip]->totalVolume - processes[ip]->upperBoundVolume;
								lowerBoundProcessId = processes[ip]->id;
							}
							if (minContributionUpperBound > processes[ip]->totalVolume - processes[ip]->lowerBoundVolume) {
								minContributionUpperBound = processes[ip]->totalVolume - processes[ip]->lowerBoundVolume;
								upperBoundProcessId = processes[ip]->id;
							}

							//if (useDelete) {
							toDelete = false;
							if (searchSubject == QEHCParameters::SearchSubjectOption::MinimumContribution) {
								if (processes[ip]->totalVolume - processes[ip]->upperBoundVolume > minContributionUpperBound) {
									toDelete = true;
								}
							}
							else {
								if (processes[ip]->totalVolume - processes[ip]->lowerBoundVolume < maxContributionLowerBound) {
									toDelete = true;
								}
							}
							if (toDelete) {
								delete processes[ip];
								processes.erase(processes.begin() + ip);
								ip--;
								break;
							}
							//}
							continue;
						}

						processes[ip]->upperBoundVolume -= subProblems[iSP].volume;

						int iPivot = subProblems[iSP].start;
						DType maxVolume;
						maxVolume = backend::Hypervolume(&subProblems[iSP].NadirPoint, ((*indexSet))[iPivot], &subProblems[iSP].IdealPoint, numberOfObjectives); //niepotrzebnie w p�tli ??

						// Find the pivot point
						unsigned i;
						for (i = subProblems[iSP].start + 1; i <= subProblems[iSP].end; i++) {
							DType volumeCurrent;
							volumeCurrent = backend::Hypervolume(&subProblems[iSP].NadirPoint, ((*indexSet))[i], &subProblems[iSP].IdealPoint, numberOfObjectives);
							if (maxVolume < volumeCurrent) {
								maxVolume = volumeCurrent;
								iPivot = i;
							}
						}

						DType v = backend::Hypervolume(&subProblems[iSP].NadirPoint, ((*indexSet))[iPivot], &subProblems[iSP].IdealPoint, numberOfObjectives);
						processes[ip]->lowerBoundVolume += v;

						if (minContributionUpperBound > processes[ip]->totalVolume - processes[ip]->lowerBoundVolume) {
							minContributionUpperBound = processes[ip]->totalVolume - processes[ip]->lowerBoundVolume;
							upperBoundProcessId = processes[ip]->id;
						}

						//if (useDelete) {
						toDelete = false;
						if (searchSubject == QEHCParameters::SearchSubjectOption::MaximumContribution) {
							if (processes[ip]->totalVolume - processes[ip]->lowerBoundVolume < maxContributionLowerBound) {
								toDelete = true;
							}
						}
						if (toDelete) {
							delete processes[ip];
							processes.erase(processes.begin() + ip);
							ip--;
							break;
						}
						//}

						processes[ip]->upperBoundVolume += v;

						unsigned j;
						int jj;
						int partStart;
						int partEnd;
						int iPos = maxIndexUsed + 1;

						Point partNadirPoint = subProblems[iSP].NadirPoint;
						Point partIdealPoint = subProblems[iSP].IdealPoint;
						Point* pPivotPoint = ((*indexSet))[iPivot];

						//------------------------------------------------------------------------------------------------
						SubproblemParam* subproblemParams = new SubproblemParam[numberOfObjectives];

						for (j = 0; j < numberOfObjectives; j++) {
							subproblemParams[j].objectiveIndex = j;
							subproblemParams[j].pointsCounter = 0;
							subproblemParams[j].partStart = iPos;

							DType lValB = partIdealPoint.ObjectiveValues[j];
							DType rValB = ((*indexSet))[iPivot]->ObjectiveValues[j];
							DType valB = lValB > rValB ? rValB : lValB;

							for (i = subProblems[iSP].start; i <= subProblems[iSP].end; i++) {
								if (i == iPivot)
									continue;
								DType lValA = partIdealPoint.ObjectiveValues[j];
								DType rValA = (*indexSet)[i]->ObjectiveValues[j];
								DType valA = lValA > rValA ? rValA : lValA;
								if (valA > valB) { 
									(*indexSet)[iPos++] = (*indexSet)[i];
									subproblemParams[j].pointsCounter++;
								}
							}

							subproblemParams[j].partEnd = iPos - 1;
							maxIndexUsed = iPos - 1;
						}

						//------------------------------------------------------------------------------------------------
						if (true) {
							std::sort(subproblemParams, subproblemParams + numberOfObjectives, sortByPointsCounterAsc);
						}
						else
						{
							if (useShuffle)
							{
								shuffle(subproblemParams, subproblemParams + numberOfObjectives, randomNumberGenerator);
							}
							else
							{
								std::rotate(subproblemParams, subproblemParams + offset, subproblemParams + numberOfObjectives);
							}
						}
						//rotate(subproblemParams, subproblemParams + offset, subproblemParams + NumberOfObjectives);

						for (jj = 0; jj < numberOfObjectives; jj++) {
							j = subproblemParams[jj].objectiveIndex;
							partStart = subproblemParams[jj].partStart;
							partEnd = subproblemParams[jj].partEnd;

							if (jj > 0) {
								int j2 = subproblemParams[jj - 1].objectiveIndex;
								partIdealPoint.ObjectiveValues[j2] = std::min(subProblems[iSP].IdealPoint.ObjectiveValues[j2],
									(*indexSet)[iPivot]->ObjectiveValues[j2]);
								partNadirPoint.ObjectiveValues[j2] = subProblems[iSP].NadirPoint.ObjectiveValues[j2];
							}

							if (partEnd >= partStart) {
								partNadirPoint.ObjectiveValues[j] = std::min(subProblems[iSP].IdealPoint.ObjectiveValues[j],
									(*indexSet)[iPivot]->ObjectiveValues[j]);

								DType v = Hypervolume(&partNadirPoint, &partIdealPoint, numberOfObjectives);
								processes[ip]->upperBoundVolume += v;

								int newISP = subProblems.getNew();
								subProblems[newISP].start = partStart;
								subProblems[newISP].end = partEnd;
								subProblems[newISP].volume = v;
								subProblems[newISP].IdealPoint = partIdealPoint;
								subProblems[newISP].NadirPoint = partNadirPoint;
								processes[ip]->subProblemsStack.push_back(newISP);
							}
						}
						delete[] subproblemParams;
						subProblems.free(iSP);

						/*if (maxIndexUsed > indexSet.size() - 10000000) {
							indexSet.resize(indexSet.size() + 10000000);
						}*/

						if (maxContributionLowerBound < processes[ip]->totalVolume - processes[ip]->upperBoundVolume) {
							maxContributionLowerBound = processes[ip]->totalVolume - processes[ip]->upperBoundVolume;
							lowerBoundProcessId = processes[ip]->id;
						}
						//if (useDelete) {
						toDelete = false;
						if (searchSubject == QEHCParameters::SearchSubjectOption::MinimumContribution) {
							if (processes[ip]->totalVolume - processes[ip]->upperBoundVolume > minContributionUpperBound) {
								toDelete = true;
							}
						}
						if (toDelete) {
							delete processes[ip];
							processes.erase(processes.begin() + ip);
							ip--;
							break;
						}

						tmpResult.MaximumContribution = maxContributionLowerBound;
						tmpResult.MinimumContribution = minContributionUpperBound;
						tmpResult.MinimumContributionIndex = upperBoundProcessId;
						tmpResult.MaximumContributionIndex = lowerBoundProcessId;


						tmpResult.type = Result::Contribution;
						//IterationCallback(ii, iterationLimit, &tmpResult);
						//}
					}

					if (iterLimit == ULONG_MAX) {
						maxIndexUsed = oldmaxIndexUsed;
					}
					iterLimit = iterationLimit;
					//for (int i = 0; i < 3; i++)
					//{
					//	std::cout << i<< ":" << processes[i]->subProblemsStack.size() << " ";
					//}
				}
			}



			//cout << "  Iterations:\t\t" << nIterations << "\n";
			QEHCResult ctrResult;
			if (searchSubject == QEHCParameters::SearchSubjectOption::MinimumContribution)
			{
				ctrResult.MinimumContribution = minContributionUpperBound;
				ctrResult.MaximumContribution = 0;
				ctrResult.MinimumContributionIndex = upperBoundProcessId;
				ctrResult.MaximumContributionIndex = -1;

			}
			else
			{
				ctrResult.MinimumContribution = 0;
				ctrResult.MaximumContribution = maxContributionLowerBound;
				ctrResult.MinimumContributionIndex = -1;
				ctrResult.MaximumContributionIndex = lowerBoundProcessId;
			}
			return ctrResult;

		}



	}
}

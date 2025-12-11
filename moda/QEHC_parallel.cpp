#include "QEHC.h"
#include "ExecutionContext.h"
#include "ExecutionService.h"
#include "ExecutionPool.h"

// this is an alternative implementation of the same algorithm for QEHC 
// this implementation is not parallelized yet
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



		void handleOnePoint(int contextId, int iterLimit, int offset, bool useShuffle, bool useSort, int numberOfObjectives, QEHCParameters::SearchSubjectOption searchSubject, int maxIndexUsed)
		{

		}

		std::tuple<bool, QEHCResult> QEHCAsyncEndpoint(int contextId, int iterLimit, int offset, bool useShuffle, bool useSort, int numberOfObjectives, QEHCParameters::SearchSubjectOption searchSubject, int maxIndexUsed) {
			DType maxContributionLowerBound = 0;
			#ifdef DBL_MAX
			DType minContributionUpperBound = DBL_MAX;
			#else
			DType minContributionUpperBound = DBL_LARGE;
			#endif
			int lowerBoundProcessId = -1;
			int upperBoundProcessId = -1;
			int ii;
			std::random_device randomDevice;
			std::mt19937 randomNumberGenerator(randomDevice());

			

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
			SubproblemsPool<SubProblem>& subProblems = *context->subProblemsPool;
			indexSet = context->points;
			ProcessData* process = context->process;
			for (ii = 0; ii < iterLimit; ii++) {

				if (process->subProblemsStack.size() == 0) {
					QEHCResult r;
					r.MaximumContribution = maxContributionLowerBound;
					r.MinimumContribution = minContributionUpperBound;
					r.MaximumContributionIndex = lowerBoundProcessId;
					r.MinimumContributionIndex = upperBoundProcessId;
					return std::tuple<bool, QEHCResult>(true, r); //to delete
				}
				int iSP = process->subProblemsStack.back();
				{
					process->subProblemsStack.pop_back();
				}
				

				if (!useShuffle && !useSort)
				{
					offset++;
					offset = offset % numberOfObjectives;
				}


				// If there is just one point
				if (subProblems[iSP].start == subProblems[iSP].end) {
					std::cout << iSP << "\n";
					DType v = Hypervolume(&subProblems[iSP].NadirPoint, ((*indexSet)[subProblems[iSP].start]), &subProblems[iSP].IdealPoint, numberOfObjectives);
					process->lowerBoundVolume += v;
					process->upperBoundVolume -= subProblems[iSP].volume - v;
					process->subProblemsStack.subProblems->free(iSP);


					if (maxContributionLowerBound < process->totalVolume - process->upperBoundVolume) {
						maxContributionLowerBound = process->totalVolume - process->upperBoundVolume;
						lowerBoundProcessId = process->id;
					}
					if (minContributionUpperBound > process->totalVolume - process->lowerBoundVolume) {
						minContributionUpperBound = process->totalVolume - process->lowerBoundVolume;
						upperBoundProcessId = process->id;
					}


					//if (useDelete) {

					if (searchSubject == QEHCParameters::SearchSubjectOption::MinimumContribution) {
						if (process->totalVolume - process->upperBoundVolume > minContributionUpperBound) {
							QEHCResult r;
							r.MaximumContribution = maxContributionLowerBound;
							r.MinimumContribution = minContributionUpperBound;
							r.MaximumContributionIndex = lowerBoundProcessId;
							r.MinimumContributionIndex = upperBoundProcessId;
							return std::tuple<bool, QEHCResult>(true, r); //to delete
						}
					}
					else {
						if (process->totalVolume - process->lowerBoundVolume < maxContributionLowerBound) {
							QEHCResult r;
							r.MaximumContribution = maxContributionLowerBound;
							r.MinimumContribution = minContributionUpperBound;
							r.MaximumContributionIndex = lowerBoundProcessId;
							r.MinimumContributionIndex = upperBoundProcessId;
							return std::tuple<bool, QEHCResult>(true, r); //to delete
						}
					}
					continue;
					//}

				}

				// If there are just two points
				if (subProblems[iSP].end - subProblems[iSP].start == 1) {
					DType v = backend::Hypervolume(&subProblems[iSP].NadirPoint, ((*indexSet)[subProblems[iSP].start]), &subProblems[iSP].IdealPoint, numberOfObjectives);
					v += backend::Hypervolume(&subProblems[iSP].NadirPoint, ((*indexSet)[subProblems[iSP].end]), &subProblems[iSP].IdealPoint, numberOfObjectives);
					Point* point2 = new Point(*(*indexSet)[subProblems[iSP].start]);
					unsigned j;
					for (j = 0; j < numberOfObjectives; j++) {

						point2->ObjectiveValues[j] = std::min(point2->ObjectiveValues[j], (*indexSet)[subProblems[iSP].end]->ObjectiveValues[j]);
					}
					v -= backend::Hypervolume(&subProblems[iSP].NadirPoint, point2, &subProblems[iSP].IdealPoint, numberOfObjectives);
					delete point2;
					process->lowerBoundVolume += v;
					process->upperBoundVolume -= subProblems[iSP].volume - v;
					process->subProblemsStack.subProblems->free(iSP);

					if (maxContributionLowerBound < process->totalVolume - process->upperBoundVolume) {
						maxContributionLowerBound = process->totalVolume - process->upperBoundVolume;
						lowerBoundProcessId = process->id;
					}
					if (minContributionUpperBound > process->totalVolume - process->lowerBoundVolume) {
						minContributionUpperBound = process->totalVolume - process->lowerBoundVolume;
						upperBoundProcessId = process->id;
					}
					//if (useDelete) {

					if (searchSubject == QEHCParameters::SearchSubjectOption::MinimumContribution) {
						if (process->totalVolume - process->upperBoundVolume > minContributionUpperBound) {
							QEHCResult r;
							r.MaximumContribution = maxContributionLowerBound;
							r.MinimumContribution = minContributionUpperBound;
							r.MaximumContributionIndex = lowerBoundProcessId;
							r.MinimumContributionIndex = upperBoundProcessId;
							return std::tuple<bool, QEHCResult>(true, r); //to delete
						}
					}
					else {
						if (process->totalVolume - process->lowerBoundVolume < maxContributionLowerBound) {
							QEHCResult r;
							r.MaximumContribution = maxContributionLowerBound;
							r.MinimumContribution = minContributionUpperBound;
							r.MaximumContributionIndex = lowerBoundProcessId;
							r.MinimumContributionIndex = upperBoundProcessId;
							return std::tuple<bool, QEHCResult>(true, r); //to delete
						}
					}

					continue;
					//}

				}

				process->upperBoundVolume -= subProblems[iSP].volume;

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
				process->lowerBoundVolume += v;

				if (minContributionUpperBound > process->totalVolume - process->lowerBoundVolume) {
					minContributionUpperBound = process->totalVolume - process->lowerBoundVolume;
					upperBoundProcessId = process->id;
				}

				//if (useDelete) {

				if (searchSubject == QEHCParameters::SearchSubjectOption::MaximumContribution) {
					if (process->totalVolume - process->lowerBoundVolume < maxContributionLowerBound) {
						QEHCResult r;
						r.MaximumContribution = maxContributionLowerBound;
						r.MinimumContribution = minContributionUpperBound;
						r.MaximumContributionIndex = lowerBoundProcessId;
						r.MinimumContributionIndex = upperBoundProcessId;
						return std::tuple<bool, QEHCResult>(true, r); //to delete
					}
				}

				//}

				process->upperBoundVolume += v;

				unsigned j;
				int jj;
				int partStart;
				int partEnd;
				int iPos = maxIndexUsed +1;

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

					if (useShuffle)
					{
						std::shuffle(subproblemParams, subproblemParams + numberOfObjectives, randomNumberGenerator);
					}
					else
					{
						std::rotate(subproblemParams, subproblemParams + offset, subproblemParams + numberOfObjectives);
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
						process->upperBoundVolume += v;

						int newISP = process->subProblemsStack.subProblems->getNew();
						subProblems[newISP].start = partStart;
						subProblems[newISP].end = partEnd;
						subProblems[newISP].volume = v;
						subProblems[newISP].IdealPoint = Point(partIdealPoint);
						subProblems[newISP].NadirPoint = Point(partNadirPoint);
						process->subProblemsStack.push_back(newISP);
					}
				}
				delete[] subproblemParams;
				process->subProblemsStack.subProblems->free(iSP);



				if (maxContributionLowerBound < process->totalVolume - process->upperBoundVolume) {
					maxContributionLowerBound = process->totalVolume - process->upperBoundVolume;
					lowerBoundProcessId = process->id;
				}

				//if (useDelete) {
				if (searchSubject == QEHCParameters::SearchSubjectOption::MinimumContribution) {
					if (process->totalVolume - process->upperBoundVolume > minContributionUpperBound) {
						QEHCResult r;
						r.MaximumContribution = maxContributionLowerBound;
						r.MinimumContribution = minContributionUpperBound;
						r.MaximumContributionIndex = lowerBoundProcessId;
						r.MinimumContributionIndex = upperBoundProcessId;
						return std::tuple<bool, QEHCResult>(true, r); //to delete
					}
				}


				//tmpResult.MaximumContribution = maxContributionLowerBound;
				//tmpResult.MinimumContribution = minContributionUpperBound;
				//tmpResult.MinimumContributionIndex = upperBoundProcessId;
				//tmpResult.MaximumContributionIndex = lowerBoundProcessId;



				//IterationCallback(ii, iterationLimit, &tmpResult);
				//}
			}
			QEHCResult r;
			r.MaximumContribution = maxContributionLowerBound;
			r.MinimumContribution = minContributionUpperBound;
			r.MaximumContributionIndex = lowerBoundProcessId;
			r.MinimumContributionIndex = upperBoundProcessId;
			return std::tuple<bool, QEHCResult>(false, r); //to delete

		}
		struct ThreadContener {
			std::future<std::tuple<bool, QEHCResult>> thread;
			int processNumber;
		};
		QEHCResult QEHC_parallel(int contextId, int numberOfSolutions, int maxlevel, QEHCParameters::SearchSubjectOption searchSubject, bool useSort, bool useShuffle, int offset, unsigned long int iterationLimit, int numberOfObjectives)
		{



			//std::shared_ptr<SubproblemsPool<SubProblem>> subProblems = std::make_shared<SubproblemsPool<SubProblem>>();
			
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

			backend::IQHVExecutionContext* context = (backend::IQHVExecutionContext*)pool->getContext(contextId);
			SubproblemsPool<SubProblem> subProblems;
			indexSet = context->points;
			int maxIndexUsed = context->maxIndexUsed;
			//if (!useDelete) {
			//	iterationLimit = ULONG_MAX;
			//}
			bool toDelete;

			Point newNadirPoint;
			int j;
			for (j = 0; j < numberOfObjectives; j++) {
				newNadirPoint.ObjectiveValues[j] = 0;
			}
			newNadirPoint.NumberOfObjectives = numberOfObjectives;

			DType maxContributionLowerBound = 0;
			DType minContributionUpperBound = 1;
			int lowerBoundProcessId = -1;
			int upperBoundProcessId = -1;

			int pos = context->initialSize;
			unsigned long int iterLimit = iterationLimit;

			int ip;
			int ip0;
			unsigned long int ii;
			std::vector<int> contexts;
			for (ip = 0; ip < numberOfSolutions; ip++) {
				ProcessData* process = new ProcessData(maxlevel);
				int newId = pool->reserveContext(numberOfSolutions, 0, numberOfObjectives, ExecutionContext::ExecutionContextType::QEHCContext, true);
				QEHCExecutionContext* newContext = (QEHCExecutionContext*)pool->getContext(newId);
				newContext->subProblemsPool = &subProblems;
				int iSP = subProblems.getNew();
				//int iSP = subProblems.getNew();
				subProblems[iSP].IdealPoint = *(*indexSet)[ip];
				subProblems[iSP].NadirPoint = newNadirPoint;
				subProblems[iSP].start = 0;//pos;
				subProblems[iSP].end = numberOfSolutions - 2;//pos + numberOfSolutions - 2;
				DType v = Hypervolume(&subProblems[iSP].NadirPoint, &subProblems[iSP].IdealPoint, numberOfObjectives);
				subProblems[iSP].volume = v;
				process->lowerBoundVolume = 0;
				process->upperBoundVolume = v;
				process->totalVolume = v;
				process->subProblemsStack.subProblems = &subProblems;
				process->subProblemsStack.push_back(iSP);
				process->id = ip;

				newContext->process = process;
				contexts.push_back(newId);

				int i3 = 0;
				int i2; for (i2 = 0; i2 < numberOfSolutions; i2++) {
					if (i2 != ip) {
						(*newContext->points)[i3++] = (*indexSet)[i2];
						
					}
				}
				pos += numberOfSolutions - 1;
			}
			maxIndexUsed = pos - 1;
			int oldmaxIndexUsed = maxIndexUsed;
			int deleted;
			std::vector<std::future<std::tuple<bool, QEHCResult>>> threads;
			std::vector<int> toRemove;
			while (contexts.size() > 0) {
				std::vector<std::tuple<int, std::tuple<bool, QEHCResult>>> results;
				std::vector<int> processIds;
				int removed = 0;
				toRemove.clear(); 
				for (ip = 0; ip < contexts.size(); ip++) {
					// -0.0180994

					auto r = QEHCAsyncEndpoint(contexts[ip], iterLimit, offset, useShuffle, useSort, numberOfObjectives, searchSubject, numberOfSolutions);
					
					
					//results.push_back(std::tuple<int, std::tuple<bool, QEHCResult>>(ip, r));

					if (std::get<0>(r))
					{
						//toRemove.push_back(ip);
						QEHCExecutionContext* currentContext = (QEHCExecutionContext*)pool->getContext(contexts[ip]);
						//processes.erase(std::remove(processes.begin(), processes.end(), currentContext->process), processes.end());
						//processes.erase(processes.begin() + ip);
						pool->releaseContext(contexts[ip]);
						contexts.erase(std::remove(contexts.begin(), contexts.end(), contexts[ip]), contexts.end());
						ip--;
						//removed++;

					}
					double __maxContributionLowerBound = std::get<1>(r).MaximumContribution;
					double __minContributionUpperBound = std::get<1>(r).MinimumContribution;
					int __lowerBoundProcessId = std::get<1>(r).MaximumContributionIndex;
					int __upperBoundProcessId = std::get<1>(r).MinimumContributionIndex;

					if (__maxContributionLowerBound > maxContributionLowerBound)
					{
						maxContributionLowerBound = __maxContributionLowerBound;
						lowerBoundProcessId = __lowerBoundProcessId;
					}
					if (__minContributionUpperBound < minContributionUpperBound)
					{
						minContributionUpperBound = __minContributionUpperBound;
						upperBoundProcessId = __upperBoundProcessId;
					}
					if (iterLimit == ULONG_MAX) {
						maxIndexUsed = oldmaxIndexUsed;
					}
					iterLimit = iterationLimit;
					//std::cout << minContributionUpperBound << "\t" << maxContributionLowerBound << "\r";
					//auto thread = std::async(QEHCAsyncEndpoint, contexts[ip], iterLimit, offset, useShuffle, useSort, numberOfObjectives, searchSubject, numberOfSolutions);
					//processIds.push_back(ip);
					//threads.push_back(thread);
					//break;
				}
				int __iter = 0;
				//for (auto& thread : threads)
				//{


					//thread.wait();
					//while(thread.wait_for(std::chrono::seconds(0)) != std::future_status::ready);
					//std::tuple<bool, QEHCResult> r = thread.get();
				//int removed = 0;
				for (auto r : results)
				{

					////SERIALIZED
					//if (std::get<0>(std::get<1>(r)))
					//{

					//	QEHCExecutionContext* currentContext = (QEHCExecutionContext*)pool->getContext(contexts[std::get<0>(r) - removed]);
					//	processes.erase(std::remove(processes.begin(), processes.end(), currentContext->process), processes.end());
					//	pool->releaseContext(contexts[std::get<0>(r) - removed]);
					//	contexts.erase(std::remove(contexts.begin(), contexts.end(), contexts[std::get<0>(r) - removed]), contexts.end());
					//	removed++;

					//}
					//double __maxContributionLowerBound = std::get<1>(std::get<1>(r)).MaximumContribution;
					//double __minContributionUpperBound = std::get<1>(std::get<1>(r)).MinimumContribution;
					//int __lowerBoundProcessId = std::get<1>(std::get<1>(r)).MaximumContributionIndex;
					//int __upperBoundProcessId = std::get<1>(std::get<1>(r)).MinimumContributionIndex;

					//PARALLEL
					//if (std::get<0>(r))
					//{
					//	QEHCExecutionContext* currentContext = (QEHCExecutionContext*)pool->getContext(contexts[processIds[__iter] - removed]);
					//	processes.erase(std::remove(processes.begin(), processes.end(), currentContext->process), processes.end());
					//	pool->releaseContext(contexts[processIds[__iter] - removed]);
					//	contexts.erase(std::remove(contexts.begin(), contexts.end(), contexts[processIds[__iter] - removed]), contexts.end());
					//	removed++;
					//	__iter++;

					//}
					//double __maxContributionLowerBound = std::get<1>(r).MaximumContribution;
					//double __minContributionUpperBound = std::get<1>(r).MinimumContribution;
					//int __lowerBoundProcessId = std::get<1>(r).MaximumContributionIndex;
					//int __upperBoundProcessId = std::get<1>(r).MinimumContributionIndex;

					//-----
					//if (__maxContributionLowerBound > maxContributionLowerBound)
					//{
					//	maxContributionLowerBound = __maxContributionLowerBound;
					//	lowerBoundProcessId = __lowerBoundProcessId;
					//}
					//if (__minContributionUpperBound < minContributionUpperBound)
					//{
					//	minContributionUpperBound = __minContributionUpperBound;
					//	upperBoundProcessId = __upperBoundProcessId;
					//}
					//if (iterLimit == ULONG_MAX) {
					//	maxIndexUsed = oldmaxIndexUsed;
					//}
					//iterLimit = iterationLimit;
				}
			}

			//processes.clear();
			for (auto contextId : contexts)
			{
				pool->releaseContext(contextId);
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


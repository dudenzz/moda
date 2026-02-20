#include "QEHC_parallel_take2.h"
#include "ExecutionContext.h"
#include "ExecutionPool.h"
#include "ExecutionService.h"
#include <mutex>

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
#if UNDERLYING_TYPE == 1
#define UType myvector<Point*>*   
#elif  UNDERLYING_TYPE == 2
#define UType SemiDynamicArray<Point*>*
#elif UNDERLYING_TYPE == 3
#define UType SecureVector<Point*>*
#elif UNDERLYING_TYPE == 4
#define UType std::vector<Point*>*
#endif
namespace moda {
	namespace backend {



		bool handleOnePoint(QEHCExecutionContext* context, Point* nadir, Point* ideal, Point* current, ProcessData* process, SubproblemsPool <SubProblem> * subProblems, int iSP,  QEHCParameters::SearchSubjectOption searchSubject)
		{
			int numberOfObjectives = context->numObjectives;
			DType v = backend::Backend::Hypervolume(nadir, current, ideal, numberOfObjectives);
			process->lowerBoundVolume += v;
			process->upperBoundVolume -= (*subProblems)[iSP].volume - v;
			subProblems->free(iSP);
			if (process->totalVolume < process->lowerBoundVolume)
			{
				int s = 1;
			}
			if (context->maxContributionLowerBound < process->totalVolume - process->upperBoundVolume) {
				context->maxContributionLowerBound = process->totalVolume - process->upperBoundVolume;
				context->lowerBoundProcessId = process->id;
			}
			if (context->minContributionUpperBound > process->totalVolume - process->lowerBoundVolume) {
				context->minContributionUpperBound = process->totalVolume - process->lowerBoundVolume;
				context->upperBoundProcessId = process->id;
			}

			//if (useDelete) {
			bool toDelete = false;
			if (searchSubject == QEHCParameters::SearchSubjectOption::MinimumContribution) {
				if (process->totalVolume - process->upperBoundVolume > context->minContributionUpperBound) {
					toDelete = true;
				}
			}
			else {
				if (process->totalVolume - process->lowerBoundVolume < context->maxContributionLowerBound) {
					toDelete = true;
				}
			}
			if (toDelete) {
				delete process;
				return true;

			}
			return false;
		}
		bool handleTwoPoints(QEHCExecutionContext* context, Point* nadir, Point* ideal, Point* p1, Point* p2, ProcessData* process, SubproblemsPool <SubProblem>* subProblems, int iSP,  QEHCParameters::SearchSubjectOption searchSubject) {
			
			DType v = backend::Backend::Hypervolume(nadir, p1, ideal, context->numObjectives);
			v += backend::Backend::Hypervolume(nadir, p2, ideal, context->numObjectives);
			process->lowerBoundVolume += v;
			process->upperBoundVolume -= (*subProblems)[iSP].volume - v;
			(*subProblems).free(iSP);

			if (context->maxContributionLowerBound < process->totalVolume - process->upperBoundVolume) {
				context->maxContributionLowerBound = process->totalVolume - process->upperBoundVolume;
				context->lowerBoundProcessId = process->id;
			}
			if (context->minContributionUpperBound > process->totalVolume - process->lowerBoundVolume) {
				context->minContributionUpperBound = process->totalVolume - process->lowerBoundVolume;
				context->upperBoundProcessId = process->id;
			}

			//if (useDelete) {
			bool toDelete = false;
			if (searchSubject == QEHCParameters::SearchSubjectOption::MinimumContribution) {
				if (process->totalVolume - process->upperBoundVolume > context->minContributionUpperBound) {
					toDelete = true;
				}
			}
			else {
				if (process->totalVolume - process->lowerBoundVolume < context->maxContributionLowerBound) {
					toDelete = true;
				}
			}
			if (toDelete) {
				delete process;
				return true;

			}
			return false;
		}
		bool handlePivot(QEHCExecutionContext* context, Point* nadir, Point* ideal, Point* pivot, ProcessData* process, SubproblemsPool <SubProblem>* subProblems, int iSP,  QEHCParameters::SearchSubjectOption searchSubject) {
			DType v = backend::Backend::Hypervolume(nadir, pivot, ideal, context->numObjectives);
			process->lowerBoundVolume += v;

			if (context->minContributionUpperBound > process->totalVolume - process->lowerBoundVolume) {
				context->minContributionUpperBound = process->totalVolume - process->lowerBoundVolume;
				context->upperBoundProcessId = process->id;
			}
			//if (useDelete) {
			bool toDelete = false;
			if (searchSubject == QEHCParameters::SearchSubjectOption::MaximumContribution) {
				if (process->totalVolume - process->lowerBoundVolume < context->maxContributionLowerBound) {
					toDelete = true;
				}
			}
			if (toDelete) {
				delete process;
				return true;
			}
			//}

			process->upperBoundVolume += v;
			return false;
		}
		SubproblemParam* buildSubProblemParams(QEHCExecutionContext* context,  int iPivot, int iSP, int& iPos, int numberOfObjectives)
		{
			SubproblemParam* subproblemParams = new SubproblemParam[numberOfObjectives];
			
			for (int j = 0; j < numberOfObjectives; j++) {
				subproblemParams[j].objectiveIndex = j;
				subproblemParams[j].pointsCounter = 0;
				subproblemParams[j].partStart = iPos;

				DType lValB = (*context->subProblemsPool)[iSP].IdealPoint.ObjectiveValues[j];
				DType rValB = (*context->points)[iPivot]->ObjectiveValues[j];
				DType valB = lValB > rValB ? rValB : lValB;

				for (int i = (*context->subProblemsPool)[iSP].start; i <= (*context->subProblemsPool)[iSP].end; i++) {
					if (i == iPivot)
						continue;
					DType lValA = (*context->subProblemsPool)[iSP].IdealPoint.ObjectiveValues[j];
					auto ptmp = (*context->points)[i];
					DType rValA = (*context->points)[i]->ObjectiveValues[j];
					DType valA = lValA > rValA ? rValA : lValA;
					if (valA > valB) {
						(*context->points)[iPos++] = (*context->points)[i];
						subproblemParams[j].pointsCounter++;
					}
				}

				subproblemParams[j].partEnd = iPos - 1;
				context->maxIndexUsed = iPos - 1;
			}
			return subproblemParams;
		}		
		int findPivot(SubproblemsPool <SubProblem>* subProblems, UType indexSet, int iSP, int numberOfObjectives)
		{
			int iPivot = (*subProblems)[iSP].start;
			DType maxVolume;
			maxVolume = backend::Backend::Hypervolume(&(*subProblems)[iSP].NadirPoint, (*indexSet)[iPivot], &(*subProblems)[iSP].IdealPoint, numberOfObjectives);

			// Find the pivot point
			unsigned i;
			for (i = (*subProblems)[iSP].start + 1; i <= (*subProblems)[iSP].end; i++) {
				DType volumeCurrent;
				volumeCurrent = backend::Backend::Hypervolume(&(*subProblems)[iSP].NadirPoint, (*indexSet)[i], &(*subProblems)[iSP].IdealPoint, numberOfObjectives);
				if (maxVolume < volumeCurrent) {
					maxVolume = volumeCurrent;
					iPivot = i;
				}
			}
			return iPivot;

		}
		void buildSubProblems(QEHCExecutionContext* context,  SubproblemParam* subproblemParams, Point* partIdealPoint, Point* partNadirPoint, ProcessData* process, int iPivot, int iSP, int numberOfObjectives)
		{
			UType indexSet = context->points;
			SubproblemsPool <SubProblem>* subProblems = context->subProblemsPool.get();
			int j, partStart, partEnd;
			for (int jj = 0; jj < numberOfObjectives; jj++) {
				j = subproblemParams[jj].objectiveIndex;
				partStart = subproblemParams[jj].partStart;
				partEnd = subproblemParams[jj].partEnd;

				if (jj > 0) {
					int j2 = subproblemParams[jj - 1].objectiveIndex;
					partIdealPoint->ObjectiveValues[j2] = std::min((*subProblems)[iSP].IdealPoint.ObjectiveValues[j2],
						(*indexSet)[iPivot]->ObjectiveValues[j2]);
					partNadirPoint->ObjectiveValues[j2] = (*subProblems)[iSP].NadirPoint.ObjectiveValues[j2];
				}

				if (partEnd >= partStart) {
					partNadirPoint->ObjectiveValues[j] = std::min((*subProblems)[iSP].IdealPoint.ObjectiveValues[j],
						(*indexSet)[iPivot]->ObjectiveValues[j]);

					DType v = Backend::Hypervolume(partNadirPoint, partIdealPoint, numberOfObjectives);
					process->upperBoundVolume += v;

					int newISP = (*subProblems).getNew();
					(*subProblems)[newISP].start = partStart;
					(*subProblems)[newISP].end = partEnd;
					(*subProblems)[newISP].volume = v;
					(*subProblems)[newISP].IdealPoint = *partIdealPoint;
					(*subProblems)[newISP].NadirPoint = *partNadirPoint;
					process->subProblemsStack.push_back(newISP);
				}
			}
		}
		bool singleQEHCIteration(QEHCExecutionContext* context, ProcessData* process, QEHCParameters::SearchSubjectOption searchSubject)
		{
			//backend::ExecutionService* poolService = &(backend::ExecutionService::getInstance());
			//backend::ExecutionPool* pool = &(poolService->getPool());
			//int newContextId = pool->reserveContext(0, 0, 0, ExecutionContext::QEHCContext, true);
			//backend::QEHCExecutionContext* newContext = (backend::QEHCExecutionContext*)pool->getContext(newContextId);
			SubproblemsPool <SubProblem>* subProblems = context->subProblemsPool.get();
			UType indexSet = context->points;
			DType maxContributionLowerBound = context->maxContributionLowerBound;
			int lowerBoundProcessId = context->lowerBoundProcessId;
			DType minContributionUpperBound = context->minContributionUpperBound;
			int upperBoundProcessId = context->maxContributionLowerBound;
			int numberOfObjectives = context->numObjectives;
			if (process->subProblemsStack.size() == 0) {
				
				return true;
			}
			int iSP = process->subProblemsStack.back();
			{
				process->subProblemsStack.pop_back();
			}
			int start = (*subProblems)[iSP].start;
			int end = (*subProblems)[iSP].end;
			// If there is just one point
			if ((*subProblems)[iSP].start == (*subProblems)[iSP].end) {
				if (handleOnePoint(
					context,
					&(*subProblems)[iSP].NadirPoint,
					&(*subProblems)[iSP].IdealPoint,
					((*indexSet)[(*subProblems)[iSP].start]),
					process,
					subProblems,
					iSP,
					searchSubject)
					)
				{
					
					return true;
				}
				else {
					return false;
				}
			}

			// If there are just two points
			if ((*subProblems)[iSP].end - (*subProblems)[iSP].start == 1) {
				if (handleTwoPoints(
					context,
					&(*subProblems)[iSP].NadirPoint,
					&(*subProblems)[iSP].IdealPoint,
					((*indexSet)[(*subProblems)[iSP].start]),
					((*indexSet)[(*subProblems)[iSP].end]),
					process,
					subProblems,
					iSP,
					searchSubject)
					)
				{
					
					return true;
				}
				else {
					return false;
				}
			}

			process->upperBoundVolume -= (*subProblems)[iSP].volume;


			int iPivot = findPivot(subProblems, &(*indexSet), iSP, numberOfObjectives);
			if (handlePivot(
				context,
				&(*subProblems)[iSP].NadirPoint,
				&(*subProblems)[iSP].IdealPoint,
				((*indexSet)[iPivot]),
				process,
				subProblems,
				iSP,
				searchSubject)
				)
			{
				
				return true;
			}
			
			//context->minContributionUpperBound = maxContributionLowerBound;
			//context->minContributionUpperBound = lowerBoundProcessId;
			//context->minContributionUpperBound = minContributionUpperBound;
			//context->upperBoundProcessId = upperBoundProcessId;
			int iPos = context->maxIndexUsed + 1;

			Point partNadirPoint = (*subProblems)[iSP].NadirPoint;
			Point partIdealPoint = (*subProblems)[iSP].IdealPoint;


			//------------------------------------------------------------------------------------------------
			SubproblemParam* subproblemParams = buildSubProblemParams(context, iPivot, iSP, iPos, numberOfObjectives);
			buildSubProblems(context, subproblemParams, &partIdealPoint, &partNadirPoint, process, iPivot, iSP, numberOfObjectives);
			delete[] subproblemParams;
			(*subProblems).free(iSP);
			return false;

		}
		QEHCResult QEHC_parallel_2(int contextId, int numberOfSolutions, int maxlevel, QEHCParameters::SearchSubjectOption searchSubject, bool useSort, bool useShuffle, int offset, unsigned long int iterationLimit, int numberOfObjectives)
		{
			QEHCResult tmpResult;
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
			context->subProblemsPool = std::make_shared< SubproblemsPool <SubProblem>>(new SubproblemsPool <SubProblem>());
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



			int pos = context->initialSize;
			unsigned long int iterLimit = iterationLimit;
			std::vector <ProcessData*> processes;
			processes.resize(numberOfSolutions);

			int ip;
			int ip0;
			unsigned long int ii;
			std::vector<QEHCExecutionContext*> contexts;
			for (ip = 0; ip < numberOfSolutions; ip++) {
				processes[ip] = new ProcessData(maxlevel);
				int iSP = context->subProblemsPool->getNew();
				(*context->subProblemsPool)[iSP].IdealPoint = *((*indexSet)[ip]);
				(*context->subProblemsPool)[iSP].NadirPoint = newNadirPoint;
				(*context->subProblemsPool)[iSP].start = pos;
				(*context->subProblemsPool)[iSP].end = pos + numberOfSolutions - 2;
				(*context->subProblemsPool)[iSP].volume = Backend::Hypervolume(&(*context->subProblemsPool)[iSP].NadirPoint, &(*context->subProblemsPool)[iSP].IdealPoint, numberOfObjectives);
				processes[ip]->lowerBoundVolume = 0;
				processes[ip]->upperBoundVolume = (*context->subProblemsPool)[iSP].volume;
				processes[ip]->totalVolume = (*context->subProblemsPool)[iSP].volume;
				processes[ip]->subProblemsStack.subProblems = &(*context->subProblemsPool);
				processes[ip]->subProblemsStack.push_back(iSP);
				processes[ip]->id = ip;
				


				int i3 = 0;
				int i2; for (i2 = 0; i2 < numberOfSolutions; i2++) {
					if (i2 != ip) {
						(*indexSet)[pos + i3++] = (*indexSet)[i2];
					}
				}

				pos += numberOfSolutions - 1;
			}
			context->maxIndexUsed = pos - 1;
			int oldmaxIndexUsed = maxIndexUsed;
			long nIterations = 0;
			bool runningPhase1 = true;
			while (processes.size() > 0) {
				for (ip = 0; ip < processes.size(); ip++) {
					for (ii = 0; ii < iterLimit; ii++) {
						if (singleQEHCIteration(context, processes[ip], searchSubject))
						{
							(processes).erase((processes).begin() + ip);
							ip--;
							break;
						}
					}
					if (iterLimit == ULONG_MAX) {
						context->maxIndexUsed = oldmaxIndexUsed;
					}
					iterLimit = iterationLimit;
				}
			}


			DType maxContributionLowerBound = context->maxContributionLowerBound;
			DType minContributionUpperBound = context->minContributionUpperBound;

			int lowerBoundProcessId = context->lowerBoundProcessId;
			int upperBoundProcessId = context->upperBoundProcessId;
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

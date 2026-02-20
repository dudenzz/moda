#include "pch.h"
#include "CppUnitTest.h"
#include "DataSet.h"
#include "IQHVSolver.h"
#include "DBHVESolver.h"
#include "Result.h"
#include "../OriginalSolutions/OriginalIQHV.h"
#include "../OriginalSolutions/OriginalHVE.h"
#include <fstream>
#define TEST_EPSILON 0.002
#define IDENTITY_EPSILON 0.000001
#define SLOWDOWN 3
#ifndef DATASET_PATH
#define DATASET_PATH "C://Users//kubad//hypervolume//hypervolume//MODA//ModaAutomatedTests//Datasets//" 
#endif
using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace moda;

namespace TimeTests
{
	TEST_CLASS(IQHV)
	{
	public:

		TEST_METHOD(TimeIQHVTest1)
		{
			auto ds = DataSet::LoadFromFilename(std::string(DATASET_PATH) + "unit_tests_d4n100_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			OriginalIQHV::NumberOfObjectives = 4;
			int nSol = 100;
			IQHVSolver solver;
			IQHVParameters* params = new IQHVParameters(SolverParameters::exact, SolverParameters::exact);

			HypervolumeResult* result2 = solver.Solve(ds, *params);
			std::vector <OriginalIQHV::TPoint*> allSolutions;
			std::ostringstream fileName;
			fileName << std::string(DATASET_PATH) + "unit_tests_d4n100_10";
			std::fstream Stream(fileName.str(), std::ios::in);
			allSolutions.clear();
			OriginalIQHV::Load(allSolutions, Stream);
			int iter = 0;
			for (auto a : ds->points)
			{
				allSolutions[iter] = new OriginalIQHV::TPoint;
				for (int objective = 0; objective <= OriginalIQHV::NumberOfObjectives; objective++)
				{
					allSolutions[iter]->ObjectiveValues[objective] = (float)a->ObjectiveValues[objective];
				}
				iter++;
			}
			Stream.close();

			OriginalIQHV::TPoint idealPoint, nadirPoint;
			int ii, jj;
			for (jj = 0; jj < OriginalIQHV::NumberOfObjectives; jj++) {
				idealPoint.ObjectiveValues[jj] = allSolutions[0]->ObjectiveValues[jj];
				nadirPoint.ObjectiveValues[jj] = allSolutions[0]->ObjectiveValues[jj];
			}
			for (ii = 1; ii < allSolutions.size(); ii++) {
				for (jj = 0; jj < OriginalIQHV::NumberOfObjectives; jj++) {
					if (idealPoint.ObjectiveValues[jj] < allSolutions[ii]->ObjectiveValues[jj]) {
						idealPoint.ObjectiveValues[jj] = allSolutions[ii]->ObjectiveValues[jj];
					}
					if (nadirPoint.ObjectiveValues[jj] > allSolutions[ii]->ObjectiveValues[jj]) {
						nadirPoint.ObjectiveValues[jj] = allSolutions[ii]->ObjectiveValues[jj];
					}
				}
			}
			//OriginalIQHV::normalize(allSolutions, idealPoint, nadirPoint);

			//for (jj = 0; jj < OriginalIQHV::NumberOfObjectives; jj++) {
			//	idealPoint.ObjectiveValues[jj] = 1;
			//	nadirPoint.ObjectiveValues[jj] = 0;
			//}
			clock_t start = clock();
			double result = OriginalIQHV::solveQHV_II(allSolutions, idealPoint, nadirPoint, nSol);
			clock_t elapsed = clock() - start;
			Assert::IsTrue(std::fabs(result2->HyperVolume - result) < IDENTITY_EPSILON);
			double current_slowdown = result2->ElapsedTime / (double)elapsed;
			Assert::IsTrue(current_slowdown < SLOWDOWN);
			Logger::WriteMessage(std::to_string(current_slowdown).c_str());
			Logger::WriteMessage(" \noriginal time: ");
			Logger::WriteMessage(std::to_string(elapsed).c_str());
			Logger::WriteMessage(" \nnew time: ");
			Logger::WriteMessage(std::to_string(result2->ElapsedTime).c_str());
		}


		TEST_METHOD(TimeIQHVTest2)
		{
			auto ds = DataSet::LoadFromFilename(std::string(DATASET_PATH) + "unit_tests_d4n200_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			OriginalIQHV::NumberOfObjectives = 4;
			int nSol = 200;
			IQHVSolver solver;
			IQHVParameters* params = new IQHVParameters(SolverParameters::exact, SolverParameters::exact);
			clock_t start1 = clock();
			ds->normalize();
			HypervolumeResult* result2 = solver.Solve(ds, *params);
			clock_t t1 = clock() - start1;
			std::vector <OriginalIQHV::TPoint*> allSolutions;
			std::ostringstream fileName;
			fileName << std::string(DATASET_PATH) + "unit_tests_d4n200_10";
			std::fstream Stream(fileName.str(), std::ios::in);
			allSolutions.clear();
			Load(allSolutions, Stream);
			int iter = 0;
			for (auto a : ds->points)
			{
				allSolutions[iter] = new OriginalIQHV::TPoint;
				for (int objective = 0; objective <= OriginalIQHV::NumberOfObjectives; objective++)
				{
					allSolutions[iter]->ObjectiveValues[objective] = (float)a->ObjectiveValues[objective];
				}
				iter++;
			}
			Stream.close();

			clock_t start2 = clock();
			OriginalIQHV::TPoint idealPoint, nadirPoint;
			int ii, jj;
			for (jj = 0; jj < OriginalIQHV::NumberOfObjectives; jj++) {
				idealPoint.ObjectiveValues[jj] = allSolutions[0]->ObjectiveValues[jj];
				nadirPoint.ObjectiveValues[jj] = allSolutions[0]->ObjectiveValues[jj];
			}
			for (ii = 1; ii < allSolutions.size(); ii++) {
				for (jj = 0; jj < OriginalIQHV::NumberOfObjectives; jj++) {
					if (idealPoint.ObjectiveValues[jj] < allSolutions[ii]->ObjectiveValues[jj]) {
						idealPoint.ObjectiveValues[jj] = allSolutions[ii]->ObjectiveValues[jj];
					}
					if (nadirPoint.ObjectiveValues[jj] > allSolutions[ii]->ObjectiveValues[jj]) {
						nadirPoint.ObjectiveValues[jj] = allSolutions[ii]->ObjectiveValues[jj];
					}
				}
			}
			OriginalIQHV::normalize(allSolutions, idealPoint, nadirPoint);

			for (jj = 0; jj < OriginalIQHV::NumberOfObjectives; jj++) {
				idealPoint.ObjectiveValues[jj] = 1;
				nadirPoint.ObjectiveValues[jj] = 0;
			}
			double result = OriginalIQHV::solveQHV_II(allSolutions, idealPoint, nadirPoint, nSol);
			clock_t t2 = clock() - start2;
			Assert::IsTrue(std::fabs(result2->HyperVolume - result) < IDENTITY_EPSILON);
			double current_slowdown = t1 / (double)t2;

			Logger::WriteMessage(std::to_string(current_slowdown).c_str());
			Logger::WriteMessage(" \noriginal time: ");
			Logger::WriteMessage(std::to_string(t2).c_str());
			Logger::WriteMessage(" \nnew time: ");
			Logger::WriteMessage(std::to_string(t1).c_str());
			Assert::IsTrue(current_slowdown < SLOWDOWN);
		}

		TEST_METHOD(TimeIQHVTest3)
		{
			auto ds = DataSet::LoadFromFilename(std::string(DATASET_PATH) + "unit_tests_d7n400_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			ds->normalize();
			OriginalIQHV::NumberOfObjectives = 7;
			int nSol = 400;
			IQHVSolver solver;
			IQHVParameters* params = new IQHVParameters(SolverParameters::exact, SolverParameters::exact);

			HypervolumeResult* result2 = solver.Solve(ds, *params);
			std::vector <OriginalIQHV::TPoint*> allSolutions;
			std::ostringstream fileName;
			fileName << std::string(DATASET_PATH) + "unit_tests_d7n400_10";
			std::fstream Stream(fileName.str(), std::ios::in);
			allSolutions.clear();
			Load(allSolutions, Stream);
			int iter = 0;
			for (auto a : ds->points)
			{
				allSolutions[iter] = new OriginalIQHV::TPoint;
				for (int objective = 0; objective <= OriginalIQHV::NumberOfObjectives; objective++)
				{
					allSolutions[iter]->ObjectiveValues[objective] = (float)a->ObjectiveValues[objective];
				}
				iter++;
			}
			Stream.close();

			OriginalIQHV::TPoint idealPoint, nadirPoint;
			int ii, jj;
			for (jj = 0; jj < OriginalIQHV::NumberOfObjectives; jj++) {
				idealPoint.ObjectiveValues[jj] = allSolutions[0]->ObjectiveValues[jj];
				nadirPoint.ObjectiveValues[jj] = allSolutions[0]->ObjectiveValues[jj];
			}
			for (ii = 1; ii < allSolutions.size(); ii++) {
				for (jj = 0; jj < OriginalIQHV::NumberOfObjectives; jj++) {
					if (idealPoint.ObjectiveValues[jj] < allSolutions[ii]->ObjectiveValues[jj]) {
						idealPoint.ObjectiveValues[jj] = allSolutions[ii]->ObjectiveValues[jj];
					}
					if (nadirPoint.ObjectiveValues[jj] > allSolutions[ii]->ObjectiveValues[jj]) {
						nadirPoint.ObjectiveValues[jj] = allSolutions[ii]->ObjectiveValues[jj];
					}
				}
			}
			OriginalIQHV::normalize(allSolutions, idealPoint, nadirPoint);

			for (jj = 0; jj < OriginalIQHV::NumberOfObjectives; jj++) {
				idealPoint.ObjectiveValues[jj] = 1;
				nadirPoint.ObjectiveValues[jj] = 0;
			}
			clock_t start = clock();
			double result = OriginalIQHV::solveQHV_II(allSolutions, idealPoint, nadirPoint, nSol);
			clock_t elapsed = clock() - start;
			Assert::IsTrue(std::fabs(result2->HyperVolume - result) < IDENTITY_EPSILON);
			double current_slowdown = result2->ElapsedTime / (double)elapsed;
			Assert::IsTrue(current_slowdown < SLOWDOWN);
			Logger::WriteMessage(std::to_string(current_slowdown).c_str());
			Logger::WriteMessage(" \noriginal time: ");
			Logger::WriteMessage(std::to_string(elapsed).c_str());
			Logger::WriteMessage(" \nnew time: ");
			Logger::WriteMessage(std::to_string(result2->ElapsedTime).c_str());
		}

		TEST_METHOD(TimeIQHVTest4)
		{
			auto ds = DataSet::LoadFromFilename(std::string(DATASET_PATH) + "unit_tests_d7n500_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			ds->normalize();
			OriginalIQHV::NumberOfObjectives = 7;
			int nSol = 500;
			IQHVSolver solver;
			IQHVParameters* params = new IQHVParameters(SolverParameters::exact, SolverParameters::exact);

			HypervolumeResult* result2 = solver.Solve(ds, *params);
			std::vector <OriginalIQHV::TPoint*> allSolutions;
			std::ostringstream fileName;
			fileName << std::string(DATASET_PATH) + "unit_tests_d7n500_10";
			std::fstream Stream(fileName.str(), std::ios::in);
			allSolutions.clear();
			Load(allSolutions, Stream);
			int iter = 0;
			for (auto a : ds->points)
			{
				allSolutions[iter] = new OriginalIQHV::TPoint;
				for (int objective = 0; objective <= OriginalIQHV::NumberOfObjectives; objective++)
				{
					allSolutions[iter]->ObjectiveValues[objective] = (float)a->ObjectiveValues[objective];
				}
				iter++;
			}
			Stream.close();

			OriginalIQHV::TPoint idealPoint, nadirPoint;
			int ii, jj;
			for (jj = 0; jj < OriginalIQHV::NumberOfObjectives; jj++) {
				idealPoint.ObjectiveValues[jj] = allSolutions[0]->ObjectiveValues[jj];
				nadirPoint.ObjectiveValues[jj] = allSolutions[0]->ObjectiveValues[jj];
			}
			for (ii = 1; ii < allSolutions.size(); ii++) {
				for (jj = 0; jj < OriginalIQHV::NumberOfObjectives; jj++) {
					if (idealPoint.ObjectiveValues[jj] < allSolutions[ii]->ObjectiveValues[jj]) {
						idealPoint.ObjectiveValues[jj] = allSolutions[ii]->ObjectiveValues[jj];
					}
					if (nadirPoint.ObjectiveValues[jj] > allSolutions[ii]->ObjectiveValues[jj]) {
						nadirPoint.ObjectiveValues[jj] = allSolutions[ii]->ObjectiveValues[jj];
					}
				}
			}
			OriginalIQHV::normalize(allSolutions, idealPoint, nadirPoint);

			for (jj = 0; jj < OriginalIQHV::NumberOfObjectives; jj++) {
				idealPoint.ObjectiveValues[jj] = 1;
				nadirPoint.ObjectiveValues[jj] = 0;
			}
			clock_t start = clock();
			double result = OriginalIQHV::solveQHV_II(allSolutions, idealPoint, nadirPoint, nSol);
			clock_t elapsed = clock() - start;
			Assert::IsTrue(std::fabs(result2->HyperVolume - result) < IDENTITY_EPSILON);
			double current_slowdown = result2->ElapsedTime / (double)elapsed;
			//Assert::IsTrue(current_slowdown < SLOWDOWN);
			Logger::WriteMessage(std::to_string(current_slowdown).c_str());
			Logger::WriteMessage(" \noriginal time: ");
			Logger::WriteMessage(std::to_string(elapsed).c_str());
			Logger::WriteMessage(" \nnew time: ");
			Logger::WriteMessage(std::to_string(result2->ElapsedTime).c_str());

		}



		TEST_METHOD(TimeHVETest1)
		{
			auto ds = DataSet::LoadFromFilename(std::string(DATASET_PATH) + "unit_tests_d4n100_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			ds->normalize();
			OriginalHVE::NumberOfObjectives = 4;
			OriginalHVE::sumDominatingFactor = pow(PI, OriginalHVE::NumberOfObjectives / 2.0) / tgamma(OriginalHVE::NumberOfObjectives / 2.0 + 1) * pow(0.5, OriginalHVE::NumberOfObjectives) /
				MONTE_CARLO_ITERATIONS;
			int seed = 22;
			int nSol = 100;
			IQHVSolver iqsolver;
			DBHVESolver solver;
			DBHVEParameters* params = new DBHVEParameters(SolverParameters::exact, SolverParameters::exact);
			IQHVParameters* iqparams = new IQHVParameters(SolverParameters::exact, SolverParameters::exact);
			params->seed = 22;
			params->MCiterations = 100000;
			HypervolumeResult* result_exact = iqsolver.Solve(ds, *iqparams);
			DBHVEResult* result2 = solver.Solve(ds, *params);
			std::vector <OriginalHVE::TPoint*> allSolutions;
			std::ostringstream fileName;
			fileName << std::string(DATASET_PATH) + "unit_tests_d4n100_10";
			std::fstream Stream(fileName.str(), std::ios::in);
			allSolutions.clear();
			OriginalHVE::Load(allSolutions, Stream);
			int iter = 0;
			for (auto a : ds->points)
			{
				allSolutions[iter] = new OriginalHVE::TPoint;
				for (int objective = 0; objective <= OriginalHVE::NumberOfObjectives; objective++)
				{
					allSolutions[iter]->ObjectiveValues[objective] = (float)a->ObjectiveValues[objective];
				}
				iter++;
			}
			Stream.close();

			OriginalHVE::TPoint idealPoint, nadirPoint;
			int ii, jj;
			for (jj = 0; jj < OriginalHVE::NumberOfObjectives; jj++) {
				idealPoint.ObjectiveValues[jj] = allSolutions[0]->ObjectiveValues[jj];
				nadirPoint.ObjectiveValues[jj] = allSolutions[0]->ObjectiveValues[jj];
			}
			for (ii = 1; ii < allSolutions.size(); ii++) {
				for (jj = 0; jj < OriginalHVE::NumberOfObjectives; jj++) {
					if (idealPoint.ObjectiveValues[jj] < allSolutions[ii]->ObjectiveValues[jj]) {
						idealPoint.ObjectiveValues[jj] = allSolutions[ii]->ObjectiveValues[jj];
					}
					if (nadirPoint.ObjectiveValues[jj] > allSolutions[ii]->ObjectiveValues[jj]) {
						nadirPoint.ObjectiveValues[jj] = allSolutions[ii]->ObjectiveValues[jj];
					}
				}
			}
			OriginalHVE::normalize(allSolutions, idealPoint, nadirPoint);

			for (jj = 0; jj < OriginalHVE::NumberOfObjectives; jj++) {
				idealPoint.ObjectiveValues[jj] = 1;
				nadirPoint.ObjectiveValues[jj] = 0;
			}
			//OriginalHVE::normalize(allSolutions, idealPoint, nadirPoint);
			clock_t start = clock();
			double result = OriginalHVE::approximateHVMax(allSolutions, idealPoint, nadirPoint, nSol,
				false, seed, true);
			clock_t elapsed = clock() - start;
			Assert::IsTrue(std::fabs(result2->HyperVolumeEstimation - result) < IDENTITY_EPSILON);
			double current_slowdown = result2->ElapsedTime / (double)elapsed;
			//Assert::IsTrue(current_slowdown < SLOWDOWN);
			Logger::WriteMessage(std::to_string(current_slowdown).c_str());
		}
	};
}
#include "pch.h"
#include "CppUnitTest.h"
#include "DataSet.h"
#include "IQHVSolver.h"
#include "DBHVESolver.h"
#include "SolverParameters.h"
#include "QEHCSolver.h"
#include "DBHVESolver.h"
#include "QHV_BQSolver.h"
#include "QHV_BR.h"
#include "MCHVESolver.h"
#include "HSSSolver.h"

#define TEST_EPSILON 0.002
#define ESTIMATION_ERROR 0.04
using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace moda;

namespace FunctionalTests
{
	TEST_CLASS(HSSTests)
	{

		TEST_METHOD(FunctionalIncHSS1)
		{
			QEHCSolver qehc_solver;
			IQHVSolver exact_solver;
			HSSSolver hss_solver;
			HSSParameters* hssparams = new HSSParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QEHCParameters* ehcparams = new QEHCParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			IQHVParameters* iqhparams = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);

			DataSet* ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n100_10");

			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			hssparams->StoppingCriteria = HSSParameters::StoppingCriteriaType::SubsetSize;
			hssparams->Strategy = HSSParameters::SubsetSelectionStrategy::Decremental;
			hssparams->CalculateHV = true;
			ehcparams->SearchSubject = QEHCParameters::SearchSubjectOption::MinimumContribution;

			//while there are 2 or more points in the dataset
			while (ds->getParameters()->nPoints > 1)
			{
				hssparams->StoppingSubsetSize = ds->getParameters()->nPoints - 1; //set HSSSolver stopping size to current size minus one
				auto subset_result = hss_solver.Solve(ds, *hssparams); //choose new subset with HSSSolver
				auto minimal_contribution = qehc_solver.Solve(ds, *ehcparams); //calculate minimal contributor with QEHCSolver
				int minimal_contributor_index = minimal_contribution->MinimumContributionIndex; //get its index
				//check if the minimal contributor is in the selected subset, if it is, the test fails
				Assert::IsTrue(std::find(subset_result->selectedPoints.begin(), subset_result->selectedPoints.end(), minimal_contributor_index) == subset_result->selectedPoints.end());
				//remove the minimal contrubutor from the dataset
				ds->remove(minimal_contributor_index);
			}
		}
		TEST_METHOD(FunctionalIncHSS2)
		{
			QEHCSolver qehc_solver;
			IQHVSolver exact_solver;
			HSSSolver hss_solver;
			HSSParameters* hssparams = new HSSParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QEHCParameters* ehcparams = new QEHCParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			IQHVParameters* iqhparams = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);

			DataSet* ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n200_10");

			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			hssparams->StoppingCriteria = HSSParameters::StoppingCriteriaType::SubsetSize;
			hssparams->Strategy = HSSParameters::SubsetSelectionStrategy::Decremental;
			ehcparams->SearchSubject = QEHCParameters::SearchSubjectOption::MinimumContribution;

			//while there are 2 or more points in the dataset
			while (ds->getParameters()->nPoints > 1)
			{
				hssparams->StoppingSubsetSize = ds->getParameters()->nPoints - 1; //set HSSSolver stopping size to current size minus one
				auto subset_result = hss_solver.Solve(ds, *hssparams); //choose new subset with HSSSolver
				auto minimal_contribution = qehc_solver.Solve(ds, *ehcparams); //calculate minimal contributor with QEHCSolver
				int minimal_contributor_index = minimal_contribution->MinimumContributionIndex; //get its index
				//check if the minimal contributor is in the selected subset, if it is, the test fails
				Assert::IsTrue(std::find(subset_result->selectedPoints.begin(), subset_result->selectedPoints.end(), minimal_contributor_index) == subset_result->selectedPoints.end());
				//remove the minimal contrubutor from the dataset
				ds->remove(minimal_contributor_index);
			}
		}
		TEST_METHOD(FunctionalIncHSS3)
		{
			QEHCSolver qehc_solver;
			IQHVSolver exact_solver;
			HSSSolver hss_solver;
			HSSParameters* hssparams = new HSSParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QEHCParameters* ehcparams = new QEHCParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			IQHVParameters* iqhparams = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);

			DataSet* ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n300_10");

			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			hssparams->StoppingCriteria = HSSParameters::StoppingCriteriaType::SubsetSize;
			hssparams->Strategy = HSSParameters::SubsetSelectionStrategy::Decremental;
			ehcparams->SearchSubject = QEHCParameters::SearchSubjectOption::MinimumContribution;

			//while there are 2 or more points in the dataset
			while (ds->getParameters()->nPoints > 1)
			{
				hssparams->StoppingSubsetSize = ds->getParameters()->nPoints - 1; //set HSSSolver stopping size to current size minus one
				auto subset_result = hss_solver.Solve(ds, *hssparams); //choose new subset with HSSSolver
				auto minimal_contribution = qehc_solver.Solve(ds, *ehcparams); //calculate minimal contributor with QEHCSolver
				int minimal_contributor_index = minimal_contribution->MinimumContributionIndex; //get its index
				//check if the minimal contributor is in the selected subset, if it is, the test fails
				Assert::IsTrue(std::find(subset_result->selectedPoints.begin(), subset_result->selectedPoints.end(), minimal_contributor_index) == subset_result->selectedPoints.end());
				//remove the minimal contrubutor from the dataset
				ds->remove(minimal_contributor_index);
			}
		}
		TEST_METHOD(FunctionalIncHSS4)
		{
			QEHCSolver qehc_solver;
			IQHVSolver exact_solver;
			HSSSolver hss_solver;
			HSSParameters* hssparams = new HSSParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QEHCParameters* ehcparams = new QEHCParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			IQHVParameters* iqhparams = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);

			DataSet* ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n400_10");

			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			hssparams->StoppingCriteria = HSSParameters::StoppingCriteriaType::SubsetSize;
			hssparams->Strategy = HSSParameters::SubsetSelectionStrategy::Decremental;
			ehcparams->SearchSubject = QEHCParameters::SearchSubjectOption::MinimumContribution;

			//while there are 2 or more points in the dataset
			while (ds->getParameters()->nPoints > 1)
			{
				hssparams->StoppingSubsetSize = ds->getParameters()->nPoints - 1; //set HSSSolver stopping size to current size minus one
				auto subset_result = hss_solver.Solve(ds, *hssparams); //choose new subset with HSSSolver
				auto minimal_contribution = qehc_solver.Solve(ds, *ehcparams); //calculate minimal contributor with QEHCSolver
				int minimal_contributor_index = minimal_contribution->MinimumContributionIndex; //get its index
				//check if the minimal contributor is in the selected subset, if it is, the test fails
				Assert::IsTrue(std::find(subset_result->selectedPoints.begin(), subset_result->selectedPoints.end(), minimal_contributor_index) == subset_result->selectedPoints.end());
				//remove the minimal contrubutor from the dataset
				ds->remove(minimal_contributor_index);
			}
		}
		TEST_METHOD(FunctionalIncHSS5)
		{
			QEHCSolver qehc_solver;
			IQHVSolver exact_solver;
			HSSSolver hss_solver;
			HSSParameters* hssparams = new HSSParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QEHCParameters* ehcparams = new QEHCParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			IQHVParameters* iqhparams = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);

			DataSet* ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n500_10");

			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			hssparams->StoppingCriteria = HSSParameters::StoppingCriteriaType::SubsetSize;
			hssparams->Strategy = HSSParameters::SubsetSelectionStrategy::Decremental;
			ehcparams->SearchSubject = QEHCParameters::SearchSubjectOption::MinimumContribution;

			//while there are 2 or more points in the dataset
			while (ds->getParameters()->nPoints > 1)
			{
				hssparams->StoppingSubsetSize = ds->getParameters()->nPoints - 1; //set HSSSolver stopping size to current size minus one
				auto subset_result = hss_solver.Solve(ds, *hssparams); //choose new subset with HSSSolver
				auto minimal_contribution = qehc_solver.Solve(ds, *ehcparams); //calculate minimal contributor with QEHCSolver
				int minimal_contributor_index = minimal_contribution->MinimumContributionIndex; //get its index
				//check if the minimal contributor is in the selected subset, if it is, the test fails
				Assert::IsTrue(std::find(subset_result->selectedPoints.begin(), subset_result->selectedPoints.end(), minimal_contributor_index) == subset_result->selectedPoints.end());
				//remove the minimal contrubutor from the dataset
				ds->remove(minimal_contributor_index);
			}
		}
		TEST_METHOD(FunctionalIncHSS6)
		{
			QEHCSolver qehc_solver;
			IQHVSolver exact_solver;
			HSSSolver hss_solver;
			HSSParameters* hssparams = new HSSParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QEHCParameters* ehcparams = new QEHCParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			IQHVParameters* iqhparams = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);

			DataSet* ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n600_10");

			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			hssparams->StoppingCriteria = HSSParameters::StoppingCriteriaType::SubsetSize;
			hssparams->Strategy = HSSParameters::SubsetSelectionStrategy::Decremental;
			ehcparams->SearchSubject = QEHCParameters::SearchSubjectOption::MinimumContribution;

			//while there are 2 or more points in the dataset
			while (ds->getParameters()->nPoints > 1)
			{
				hssparams->StoppingSubsetSize = ds->getParameters()->nPoints - 1; //set HSSSolver stopping size to current size minus one
				auto subset_result = hss_solver.Solve(ds, *hssparams); //choose new subset with HSSSolver
				auto minimal_contribution = qehc_solver.Solve(ds, *ehcparams); //calculate minimal contributor with QEHCSolver
				int minimal_contributor_index = minimal_contribution->MinimumContributionIndex; //get its index
				//check if the minimal contributor is in the selected subset, if it is, the test fails
				Assert::IsTrue(std::find(subset_result->selectedPoints.begin(), subset_result->selectedPoints.end(), minimal_contributor_index) == subset_result->selectedPoints.end());
				//remove the minimal contrubutor from the dataset
				ds->remove(minimal_contributor_index);
			}
		}

		TEST_METHOD(FunctionalIncHSS7)
		{
			QEHCSolver qehc_solver;
			IQHVSolver exact_solver;
			HSSSolver hss_solver;
			HSSParameters* hssparams = new HSSParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QEHCParameters* ehcparams = new QEHCParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			IQHVParameters* iqhparams = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);

			DataSet* ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n600_10");

			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			hssparams->StoppingCriteria = HSSParameters::StoppingCriteriaType::SubsetSize;
			hssparams->Strategy = HSSParameters::SubsetSelectionStrategy::Decremental;
			ehcparams->SearchSubject = QEHCParameters::SearchSubjectOption::MinimumContribution;

			//while there are 2 or more points in the dataset
			while (ds->getParameters()->nPoints > 1)
			{
				hssparams->StoppingSubsetSize = ds->getParameters()->nPoints - 1; //set HSSSolver stopping size to current size minus one
				auto subset_result = hss_solver.Solve(ds, *hssparams); //choose new subset with HSSSolver
				auto minimal_contribution = qehc_solver.Solve(ds, *ehcparams); //calculate minimal contributor with QEHCSolver
				int minimal_contributor_index = minimal_contribution->MinimumContributionIndex; //get its index
				//check if the minimal contributor is in the selected subset, if it is, the test fails
				Assert::IsTrue(std::find(subset_result->selectedPoints.begin(), subset_result->selectedPoints.end(), minimal_contributor_index) == subset_result->selectedPoints.end());
				//remove the minimal contrubutor from the dataset
				ds->remove(minimal_contributor_index);
			}
		}
		TEST_METHOD(FunctionalIncHSS8)
		{
			QEHCSolver qehc_solver;
			IQHVSolver exact_solver;
			HSSSolver hss_solver;
			HSSParameters* hssparams = new HSSParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QEHCParameters* ehcparams = new QEHCParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			IQHVParameters* iqhparams = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);

			DataSet* ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n600_10");

			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			hssparams->StoppingCriteria = HSSParameters::StoppingCriteriaType::SubsetSize;
			hssparams->Strategy = HSSParameters::SubsetSelectionStrategy::Decremental;
			ehcparams->SearchSubject = QEHCParameters::SearchSubjectOption::MinimumContribution;

			//while there are 2 or more points in the dataset
			while (ds->getParameters()->nPoints > 1)
			{
				hssparams->StoppingSubsetSize = ds->getParameters()->nPoints - 1; //set HSSSolver stopping size to current size minus one
				auto subset_result = hss_solver.Solve(ds, *hssparams); //choose new subset with HSSSolver
				auto minimal_contribution = qehc_solver.Solve(ds, *ehcparams); //calculate minimal contributor with QEHCSolver
				int minimal_contributor_index = minimal_contribution->MinimumContributionIndex; //get its index
				//check if the minimal contributor is in the selected subset, if it is, the test fails
				Assert::IsTrue(std::find(subset_result->selectedPoints.begin(), subset_result->selectedPoints.end(), minimal_contributor_index) == subset_result->selectedPoints.end());
				//remove the minimal contrubutor from the dataset
				ds->remove(minimal_contributor_index);
			}
		}
		TEST_METHOD(FunctionalIncHSS9)
		{
			QEHCSolver qehc_solver;
			IQHVSolver exact_solver;
			HSSSolver hss_solver;
			HSSParameters* hssparams = new HSSParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QEHCParameters* ehcparams = new QEHCParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			IQHVParameters* iqhparams = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);

			DataSet* ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n600_10");

			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			hssparams->StoppingCriteria = HSSParameters::StoppingCriteriaType::SubsetSize;
			hssparams->Strategy = HSSParameters::SubsetSelectionStrategy::Decremental;
			ehcparams->SearchSubject = QEHCParameters::SearchSubjectOption::MinimumContribution;

			//while there are 2 or more points in the dataset
			while (ds->getParameters()->nPoints > 1)
			{
				hssparams->StoppingSubsetSize = ds->getParameters()->nPoints - 1; //set HSSSolver stopping size to current size minus one
				auto subset_result = hss_solver.Solve(ds, *hssparams); //choose new subset with HSSSolver
				auto minimal_contribution = qehc_solver.Solve(ds, *ehcparams); //calculate minimal contributor with QEHCSolver
				int minimal_contributor_index = minimal_contribution->MinimumContributionIndex; //get its index
				//check if the minimal contributor is in the selected subset, if it is, the test fails
				Assert::IsTrue(std::find(subset_result->selectedPoints.begin(), subset_result->selectedPoints.end(), minimal_contributor_index) == subset_result->selectedPoints.end());
				//remove the minimal contrubutor from the dataset
				ds->remove(minimal_contributor_index);
			}
		}
		TEST_METHOD(FunctionalIncHSS10)
		{
			QEHCSolver qehc_solver;
			IQHVSolver exact_solver;
			HSSSolver hss_solver;
			HSSParameters* hssparams = new HSSParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QEHCParameters* ehcparams = new QEHCParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			IQHVParameters* iqhparams = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);

			DataSet* ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n600_10");

			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			hssparams->StoppingCriteria = HSSParameters::StoppingCriteriaType::SubsetSize;
			hssparams->Strategy = HSSParameters::SubsetSelectionStrategy::Decremental;
			ehcparams->SearchSubject = QEHCParameters::SearchSubjectOption::MinimumContribution;

			//while there are 2 or more points in the dataset
			while (ds->getParameters()->nPoints > 1)
			{
				hssparams->StoppingSubsetSize = ds->getParameters()->nPoints - 1; //set HSSSolver stopping size to current size minus one
				auto subset_result = hss_solver.Solve(ds, *hssparams); //choose new subset with HSSSolver
				auto minimal_contribution = qehc_solver.Solve(ds, *ehcparams); //calculate minimal contributor with QEHCSolver
				int minimal_contributor_index = minimal_contribution->MinimumContributionIndex; //get its index
				//check if the minimal contributor is in the selected subset, if it is, the test fails
				Assert::IsTrue(std::find(subset_result->selectedPoints.begin(), subset_result->selectedPoints.end(), minimal_contributor_index) == subset_result->selectedPoints.end());
				//remove the minimal contrubutor from the dataset
				ds->remove(minimal_contributor_index);
			}
		}
		TEST_METHOD(FunctionalIncHSS11)
		{
			QEHCSolver qehc_solver;
			IQHVSolver exact_solver;
			HSSSolver hss_solver;
			HSSParameters* hssparams = new HSSParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QEHCParameters* ehcparams = new QEHCParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			IQHVParameters* iqhparams = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);

			DataSet* ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n600_10");

			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			hssparams->StoppingCriteria = HSSParameters::StoppingCriteriaType::SubsetSize;
			hssparams->Strategy = HSSParameters::SubsetSelectionStrategy::Decremental;
			ehcparams->SearchSubject = QEHCParameters::SearchSubjectOption::MinimumContribution;

			//while there are 2 or more points in the dataset
			while (ds->getParameters()->nPoints > 1)
			{
				hssparams->StoppingSubsetSize = ds->getParameters()->nPoints - 1; //set HSSSolver stopping size to current size minus one
				auto subset_result = hss_solver.Solve(ds, *hssparams); //choose new subset with HSSSolver
				auto minimal_contribution = qehc_solver.Solve(ds, *ehcparams); //calculate minimal contributor with QEHCSolver
				int minimal_contributor_index = minimal_contribution->MinimumContributionIndex; //get its index
				//check if the minimal contributor is in the selected subset, if it is, the test fails
				Assert::IsTrue(std::find(subset_result->selectedPoints.begin(), subset_result->selectedPoints.end(), minimal_contributor_index) == subset_result->selectedPoints.end());
				//remove the minimal contrubutor from the dataset
				ds->remove(minimal_contributor_index);
			}
		}
		TEST_METHOD(FunctionalIncHSS12)
		{
			QEHCSolver qehc_solver;
			IQHVSolver exact_solver;
			HSSSolver hss_solver;
			HSSParameters* hssparams = new HSSParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QEHCParameters* ehcparams = new QEHCParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			IQHVParameters* iqhparams = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);

			DataSet* ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n600_10");

			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			hssparams->StoppingCriteria = HSSParameters::StoppingCriteriaType::SubsetSize;
			hssparams->Strategy = HSSParameters::SubsetSelectionStrategy::Decremental;
			ehcparams->SearchSubject = QEHCParameters::SearchSubjectOption::MinimumContribution;

			//while there are 2 or more points in the dataset
			while (ds->getParameters()->nPoints > 1)
			{
				hssparams->StoppingSubsetSize = ds->getParameters()->nPoints - 1; //set HSSSolver stopping size to current size minus one
				auto subset_result = hss_solver.Solve(ds, *hssparams); //choose new subset with HSSSolver
				auto minimal_contribution = qehc_solver.Solve(ds, *ehcparams); //calculate minimal contributor with QEHCSolver
				int minimal_contributor_index = minimal_contribution->MinimumContributionIndex; //get its index
				//check if the minimal contributor is in the selected subset, if it is, the test fails
				Assert::IsTrue(std::find(subset_result->selectedPoints.begin(), subset_result->selectedPoints.end(), minimal_contributor_index) == subset_result->selectedPoints.end());
				//remove the minimal contrubutor from the dataset
				ds->remove(minimal_contributor_index);
			}
		}
	};
	TEST_CLASS(MCHVTests)
	{
	public:
		TEST_METHOD(FunctionalMCHV1)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n500_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			MCHVESolver estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			MCHVParameters* params = new MCHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolumeEstimation - exact_result->HyperVolume) < ESTIMATION_ERROR);
		}
		TEST_METHOD(FunctionalMCHV2)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n400_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			MCHVESolver estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			MCHVParameters* params = new MCHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolumeEstimation - exact_result->HyperVolume) < ESTIMATION_ERROR);
		}
		TEST_METHOD(FunctionalMCHV3)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n300_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			MCHVESolver estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			MCHVParameters* params = new MCHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolumeEstimation - exact_result->HyperVolume) < ESTIMATION_ERROR);
		}
		TEST_METHOD(FunctionalMCHV4)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n200_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			MCHVESolver estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			MCHVParameters* params = new MCHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolumeEstimation - exact_result->HyperVolume) < ESTIMATION_ERROR);
		}
		TEST_METHOD(FunctionalMCHV5)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n100_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			MCHVESolver estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			MCHVParameters* params = new MCHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolumeEstimation - exact_result->HyperVolume) < ESTIMATION_ERROR);
		}
		TEST_METHOD(FunctionalMCHV6)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n600_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			MCHVESolver estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			MCHVParameters* params = new MCHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolumeEstimation - exact_result->HyperVolume) < ESTIMATION_ERROR);
		}
		TEST_METHOD(FunctionalMCHV7)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d6n100_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			MCHVESolver estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			MCHVParameters* params = new MCHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolumeEstimation - exact_result->HyperVolume) < ESTIMATION_ERROR);
		}
		TEST_METHOD(FunctionalMCHV8)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d6n200_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			MCHVESolver estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			MCHVParameters* params = new MCHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolumeEstimation - exact_result->HyperVolume) < ESTIMATION_ERROR);
		}
		TEST_METHOD(FunctionalMCHV9)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d6n300_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			MCHVESolver estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			MCHVParameters* params = new MCHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolumeEstimation - exact_result->HyperVolume) < ESTIMATION_ERROR);
		}
		TEST_METHOD(FunctionalMCHV10)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d6n400_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			MCHVESolver estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			MCHVParameters* params = new MCHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolumeEstimation - exact_result->HyperVolume) < ESTIMATION_ERROR);
		}
		TEST_METHOD(FunctionalMCHV11)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d6n500_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			MCHVESolver estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			MCHVParameters* params = new MCHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolumeEstimation - exact_result->HyperVolume) < ESTIMATION_ERROR);
		}
		TEST_METHOD(FunctionalMCHV12)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d6n600_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			MCHVESolver estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			MCHVParameters* params = new MCHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolumeEstimation - exact_result->HyperVolume) < ESTIMATION_ERROR);
		}
		TEST_METHOD(FunctionalMCHV13)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d7n200_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			MCHVESolver estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			MCHVParameters* params = new MCHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolumeEstimation - exact_result->HyperVolume) < ESTIMATION_ERROR);
		}
		TEST_METHOD(FunctionalMCHV14)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d7n300_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			MCHVESolver estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			MCHVParameters* params = new MCHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolumeEstimation - exact_result->HyperVolume) < ESTIMATION_ERROR);
		}
		TEST_METHOD(FunctionalMCHV15)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d7n400_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			MCHVESolver estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			MCHVParameters* params = new MCHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolumeEstimation - exact_result->HyperVolume) < ESTIMATION_ERROR);
		}
		TEST_METHOD(FunctionalMCHV16)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d7n500_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			MCHVESolver estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			MCHVParameters* params = new MCHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolumeEstimation - exact_result->HyperVolume) < ESTIMATION_ERROR);
		}
	};
	TEST_CLASS(QHV_BQ_TESTS)
	{
		TEST_METHOD(FunctionalQHV_BQ1)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n500_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			QHV_BQSolver estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QHV_BQParameters* params = new QHV_BQParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolumeEstimation - exact_result->HyperVolume) < ESTIMATION_ERROR);
		}
		TEST_METHOD(FunctionalQHV_BQ2)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n400_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			QHV_BQSolver estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QHV_BQParameters* params = new QHV_BQParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolumeEstimation - exact_result->HyperVolume) < ESTIMATION_ERROR);
		}
		TEST_METHOD(FunctionalQHV_BQ3)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n300_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			QHV_BQSolver estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QHV_BQParameters* params = new QHV_BQParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolumeEstimation - exact_result->HyperVolume) < ESTIMATION_ERROR);
		}
		TEST_METHOD(FunctionalQHV_BQ4)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n200_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			QHV_BQSolver estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QHV_BQParameters* params = new QHV_BQParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolumeEstimation - exact_result->HyperVolume) < ESTIMATION_ERROR);
		}
		TEST_METHOD(FunctionalQHV_BQ5)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n100_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			QHV_BQSolver estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QHV_BQParameters* params = new QHV_BQParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolumeEstimation - exact_result->HyperVolume) < ESTIMATION_ERROR);
		}
		TEST_METHOD(FunctionalQHV_BQ6)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n600_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			QHV_BQSolver estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QHV_BQParameters* params = new QHV_BQParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolumeEstimation - exact_result->HyperVolume) < ESTIMATION_ERROR);
		}
		TEST_METHOD(FunctionalQHV_BQ7)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d6n100_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			QHV_BQSolver estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QHV_BQParameters* params = new QHV_BQParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolumeEstimation - exact_result->HyperVolume) < ESTIMATION_ERROR);
		}
		TEST_METHOD(FunctionalQHV_BQ8)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d6n200_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			QHV_BQSolver estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QHV_BQParameters* params = new QHV_BQParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolumeEstimation - exact_result->HyperVolume) < ESTIMATION_ERROR);
		}
		TEST_METHOD(FunctionalQHV_BQ9)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d6n300_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			QHV_BQSolver estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QHV_BQParameters* params = new QHV_BQParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolumeEstimation - exact_result->HyperVolume) < ESTIMATION_ERROR);
		}
		TEST_METHOD(FunctionalQHV_BQ10)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d6n400_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			QHV_BQSolver estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QHV_BQParameters* params = new QHV_BQParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolumeEstimation - exact_result->HyperVolume) < ESTIMATION_ERROR);
		}
		TEST_METHOD(FunctionalQHV_BQ11)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d6n500_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			QHV_BQSolver estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QHV_BQParameters* params = new QHV_BQParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolumeEstimation - exact_result->HyperVolume) < ESTIMATION_ERROR);
		}
		TEST_METHOD(FunctionalQHV_BQ12)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d6n600_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			QHV_BQSolver estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QHV_BQParameters* params = new QHV_BQParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolumeEstimation - exact_result->HyperVolume) < ESTIMATION_ERROR);
		}
		TEST_METHOD(FunctionalQHV_BQ13)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d7n200_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			QHV_BQSolver estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QHV_BQParameters* params = new QHV_BQParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolumeEstimation - exact_result->HyperVolume) < ESTIMATION_ERROR);
		}
		TEST_METHOD(FunctionalQHV_BQ14)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d7n300_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			QHV_BQSolver estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QHV_BQParameters* params = new QHV_BQParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolumeEstimation - exact_result->HyperVolume) < ESTIMATION_ERROR);
		}
		TEST_METHOD(FunctionalQHV_BQ15)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d7n400_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			QHV_BQSolver estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QHV_BQParameters* params = new QHV_BQParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolumeEstimation - exact_result->HyperVolume) < ESTIMATION_ERROR);
		}
		TEST_METHOD(FunctionalQHV_BQ16)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d7n500_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			QHV_BQSolver estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QHV_BQParameters* params = new QHV_BQParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolumeEstimation - exact_result->HyperVolume) < ESTIMATION_ERROR);
		}
	};
	TEST_CLASS(QHV_BR_TESTS)
	{
		TEST_METHOD(FunctionalQHV_BR1)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n500_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			QHV_BR estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QHV_BRParameters* params = new QHV_BRParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(result->UpperBound - exact_result->HyperVolume >= 0);
			Assert::IsTrue(result->LowerBound - exact_result->HyperVolume <= 0);
		}
		TEST_METHOD(FunctionalQHV_BR2)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n400_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			QHV_BR estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QHV_BRParameters* params = new QHV_BRParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(result->UpperBound - exact_result->HyperVolume >= 0);
			Assert::IsTrue(result->LowerBound - exact_result->HyperVolume <= 0);
		}
		TEST_METHOD(FunctionalQHV_BR3)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n300_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			QHV_BR estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QHV_BRParameters* params = new QHV_BRParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(result->UpperBound - exact_result->HyperVolume >= 0);
			Assert::IsTrue(result->LowerBound - exact_result->HyperVolume <= 0);
		}
		TEST_METHOD(FunctionalQHV_BR4)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n200_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			QHV_BR estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QHV_BRParameters* params = new QHV_BRParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(result->UpperBound - exact_result->HyperVolume >= 0);
			Assert::IsTrue(result->LowerBound - exact_result->HyperVolume <= 0);
		}
		TEST_METHOD(FunctionalQHV_BR5)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n100_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			QHV_BR estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QHV_BRParameters* params = new QHV_BRParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(result->UpperBound - exact_result->HyperVolume >= 0);
			Assert::IsTrue(result->LowerBound - exact_result->HyperVolume <= 0);
		}
		TEST_METHOD(FunctionalQHV_BR6)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n600_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			QHV_BR estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QHV_BRParameters* params = new QHV_BRParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(result->UpperBound - exact_result->HyperVolume >= 0);
			Assert::IsTrue(result->LowerBound - exact_result->HyperVolume <= 0);
		}
		TEST_METHOD(FunctionalQHV_BR7)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d6n100_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			QHV_BR estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QHV_BRParameters* params = new QHV_BRParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(result->UpperBound - exact_result->HyperVolume >= 0);
			Assert::IsTrue(result->LowerBound - exact_result->HyperVolume <= 0);
		}
		TEST_METHOD(FunctionalQHV_BR8)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d6n200_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			QHV_BR estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QHV_BRParameters* params = new QHV_BRParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(result->UpperBound - exact_result->HyperVolume >= 0);
			Assert::IsTrue(result->LowerBound - exact_result->HyperVolume <= 0);
		}
		TEST_METHOD(FunctionalQHV_BR9)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d6n300_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			QHV_BR estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QHV_BRParameters* params = new QHV_BRParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(result->UpperBound - exact_result->HyperVolume >= 0);
			Assert::IsTrue(result->LowerBound - exact_result->HyperVolume <= 0);
		}
		TEST_METHOD(FunctionalQHV_BR10)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d6n400_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			QHV_BR estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QHV_BRParameters* params = new QHV_BRParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(result->UpperBound - exact_result->HyperVolume >= 0);
			Assert::IsTrue(result->LowerBound - exact_result->HyperVolume <= 0);
		}
		TEST_METHOD(FunctionalQHV_BR11)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d6n500_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			QHV_BR estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QHV_BRParameters* params = new QHV_BRParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(result->UpperBound - exact_result->HyperVolume >= 0);
			Assert::IsTrue(result->LowerBound - exact_result->HyperVolume <= 0);
		}
		TEST_METHOD(FunctionalQHV_BR12)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d6n600_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			QHV_BR estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QHV_BRParameters* params = new QHV_BRParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(result->UpperBound - exact_result->HyperVolume >= 0);
			Assert::IsTrue(result->LowerBound - exact_result->HyperVolume <= 0);
		}
		TEST_METHOD(FunctionalQHV_BR13)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d7n200_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			QHV_BR estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QHV_BRParameters* params = new QHV_BRParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(result->UpperBound - exact_result->HyperVolume >= 0);
			Assert::IsTrue(result->LowerBound - exact_result->HyperVolume <= 0);
		}
		TEST_METHOD(FunctionalQHV_BR14)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d7n300_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			QHV_BR estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QHV_BRParameters* params = new QHV_BRParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(result->UpperBound - exact_result->HyperVolume >= 0);
			Assert::IsTrue(result->LowerBound - exact_result->HyperVolume <= 0);
		}
		TEST_METHOD(FunctionalQHV_BR15)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d7n400_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			QHV_BR estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QHV_BRParameters* params = new QHV_BRParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(result->UpperBound - exact_result->HyperVolume >= 0);
			Assert::IsTrue(result->LowerBound - exact_result->HyperVolume <= 0);
		}
		TEST_METHOD(FunctionalQHV_BR16)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d7n500_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			//ds->normalize();
			//ds->setNormalize(true);

			QHV_BR estimation_solver;
			IQHVSolver exact_solver;
			IQHVParameters* parmiq = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			QHV_BRParameters* params = new QHV_BRParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			auto exact_result = exact_solver.Solve(ds, *parmiq);
			BoundedResult* result = estimation_solver.Solve(ds, *params);
			Assert::IsTrue(result->UpperBound - exact_result->HyperVolume >= 0);
			Assert::IsTrue(result->LowerBound - exact_result->HyperVolume <= 0);
		}
	};
	TEST_CLASS(IQHVTests)
	{
	public:
		TEST_METHOD(FunctionalIQHV1)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n100_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			ds->normalize();
			//ds->setNormalize(true);

			IQHVSolver solver;
			IQHVParameters* params = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);

			HypervolumeResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolume - 0.5318) < TEST_EPSILON);
		}

		TEST_METHOD(FunctionalIQHV2)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n200_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			ds->normalize();

			IQHVSolver solver;
			IQHVParameters* params = new IQHVParameters(SolverParameters::exact, SolverParameters::exact);

			HypervolumeResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolume - 0.6051) < TEST_EPSILON);
		}

		TEST_METHOD(FunctionalIQHV3)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n300_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			ds->normalize();

			IQHVSolver solver;
			IQHVParameters* params = new IQHVParameters(SolverParameters::exact, SolverParameters::exact);

			HypervolumeResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolume - 0.7676) < TEST_EPSILON);

		}

		TEST_METHOD(FunctionalIQHV4)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n400_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			ds->normalize();


			IQHVSolver solver;
			IQHVParameters* params = new IQHVParameters(SolverParameters::exact, SolverParameters::exact);

			HypervolumeResult* result = solver.Solve(ds, *params);

			Assert::IsTrue(std::fabs(result->HyperVolume - 0.7323) < TEST_EPSILON);

		}

		TEST_METHOD(FunctionalIQHV5)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n500_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			ds->normalize();

			IQHVSolver solver;
			IQHVParameters* params = new IQHVParameters(SolverParameters::exact, SolverParameters::exact);

			HypervolumeResult* result = solver.Solve(ds, *params);

			Assert::IsTrue(std::fabs(result->HyperVolume - 0.7420) < TEST_EPSILON);
		}

		TEST_METHOD(FunctionalIQHV6)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n600_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			ds->normalize();

			IQHVSolver solver;
			IQHVParameters* params = new IQHVParameters(SolverParameters::exact, SolverParameters::exact);

			HypervolumeResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolume - 0.7869) < TEST_EPSILON);
		}
		TEST_METHOD(FunctionalIQHV7)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d6n100_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			ds->normalize();

			IQHVSolver solver;
			IQHVParameters* params = new IQHVParameters(SolverParameters::exact, SolverParameters::exact);

			HypervolumeResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolume - 0.2612) < TEST_EPSILON);
		}
		TEST_METHOD(FunctionalIQHV8)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d6n200_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			ds->normalize();

			IQHVSolver solver;
			IQHVParameters* params = new IQHVParameters(SolverParameters::exact, SolverParameters::exact);

			HypervolumeResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolume - 0.3127) < TEST_EPSILON);
		}
		TEST_METHOD(FunctionalIQHV9)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d6n300_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			ds->normalize();

			IQHVSolver solver;
			IQHVParameters* params = new IQHVParameters(SolverParameters::exact, SolverParameters::exact);

			HypervolumeResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolume - 0.3618) < TEST_EPSILON);
		}
		TEST_METHOD(FunctionalIQHV10)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d6n400_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			ds->normalize();

			IQHVSolver solver;
			IQHVParameters* params = new IQHVParameters(SolverParameters::exact, SolverParameters::exact);

			HypervolumeResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolume - 0.4798) < TEST_EPSILON);
		}
		TEST_METHOD(FunctionalIQHV11)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d6n500_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			ds->normalize();

			IQHVSolver solver;
			IQHVParameters* params = new IQHVParameters(SolverParameters::exact, SolverParameters::exact);

			HypervolumeResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolume - 0.4582) < TEST_EPSILON);
		}
		TEST_METHOD(FunctionalIQHV12)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d6n600_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			ds->normalize();

			IQHVSolver solver;
			IQHVParameters* params = new IQHVParameters(SolverParameters::exact, SolverParameters::exact);

			HypervolumeResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolume - 0.4796) < TEST_EPSILON);
		}
		TEST_METHOD(FunctionalIQHV13)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d7n200_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			ds->normalize();

			IQHVSolver solver;
			IQHVParameters* params = new IQHVParameters(SolverParameters::exact, SolverParameters::exact);

			HypervolumeResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolume - 0.0332) < TEST_EPSILON);
		}
		TEST_METHOD(FunctionalIQHV14)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d7n300_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			ds->normalize();

			IQHVSolver solver;
			IQHVParameters* params = new IQHVParameters(SolverParameters::exact, SolverParameters::exact);

			HypervolumeResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolume - 0.0243) < TEST_EPSILON);
		}
		TEST_METHOD(FunctionalIQHV15)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d7n400_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			ds->normalize();

			IQHVSolver solver;
			IQHVParameters* params = new IQHVParameters(SolverParameters::exact, SolverParameters::exact);

			HypervolumeResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolume - 0.0269) < TEST_EPSILON);
		}
		TEST_METHOD(FunctionalIQHV16)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d7n500_10");
			ds->typeOfOptimization = DataSet::OptimizationType::maximization;
			ds->normalize();

			IQHVSolver solver;
			IQHVParameters* params = new IQHVParameters(SolverParameters::exact, SolverParameters::exact);

			HypervolumeResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(std::fabs(result->HyperVolume - 0.6466) < TEST_EPSILON);
		}
	};
	TEST_CLASS(QEHCTests)
	{
		TEST_METHOD(FunctionalQEHCmin1)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n100_10");
			//ds->normalize();
			QEHCSolver solver;
			QEHCParameters* params = new QEHCParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			params->SearchSubject = QEHCParameters::SearchSubjectOption::MinimumContribution;
			QEHCResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(result->MinimumContribution > 0.000005085);
			Assert::IsTrue(result->MinimumContribution < 0.000005095);
		}

		TEST_METHOD(FunctionalQEHCmin2)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n200_10");
			QEHCSolver solver;
			QEHCParameters* params = new QEHCParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			params->SearchSubject = QEHCParameters::SearchSubjectOption::MinimumContribution;
			QEHCResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(result->MinimumContribution > 0.00000173);
			Assert::IsTrue(result->MinimumContribution < 0.00000174);
		}

		TEST_METHOD(FunctionalQEHCmin3)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n300_10");
			QEHCSolver solver;
			QEHCParameters* params = new QEHCParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			params->SearchSubject = QEHCParameters::SearchSubjectOption::MinimumContribution;
			QEHCResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(result->MinimumContribution > 0.000000485);
			Assert::IsTrue(result->MinimumContribution < 0.000000495);
		}

		TEST_METHOD(FunctionalQEHCmin4)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n400_10");
			QEHCSolver solver;
			QEHCParameters* params = new QEHCParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			params->SearchSubject = QEHCParameters::SearchSubjectOption::MinimumContribution;
			QEHCResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(result->MinimumContribution > 0.000000430);
			Assert::IsTrue(result->MinimumContribution < 0.000000440);
		}

		TEST_METHOD(FunctionalQEHCmin5)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n500_10");
			QEHCSolver solver;
			QEHCParameters* params = new QEHCParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			params->SearchSubject = QEHCParameters::SearchSubjectOption::MinimumContribution;
			QEHCResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(result->MinimumContribution > 0.00000008505);
			Assert::IsTrue(result->MinimumContribution < 0.00000008515);
		}

		TEST_METHOD(FunctionalQEHCmin6)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n600_10");
			QEHCSolver solver;
			QEHCParameters* params = new QEHCParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			params->SearchSubject = QEHCParameters::SearchSubjectOption::MinimumContribution;
			params->sort = false;
			params->shuffle = false;
			QEHCResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(result->MinimumContribution > 0.0000001844);
			Assert::IsTrue(result->MinimumContribution < 0.0000001846);
		}
		TEST_METHOD(FunctionalQEHCmin7)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d6n100_10");
			QEHCSolver solver;
			QEHCParameters* params = new QEHCParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			params->SearchSubject = QEHCParameters::SearchSubjectOption::MinimumContribution;
			params->sort = false;
			params->shuffle = false;
			QEHCResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(result->MinimumContribution > 0.0000186403);
			Assert::IsTrue(result->MinimumContribution < 0.0000186404);
		}
		TEST_METHOD(FunctionalQEHCmin8)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d6n200_10");
			QEHCSolver solver;
			QEHCParameters* params = new QEHCParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			params->SearchSubject = QEHCParameters::SearchSubjectOption::MinimumContribution;
			params->sort = false;
			params->shuffle = false;
			QEHCResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(result->MinimumContribution > 0.00000605);
			Assert::IsTrue(result->MinimumContribution < 0.00000606);
		}
		TEST_METHOD(FunctionalQEHCmin9)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d6n300_10");
			QEHCSolver solver;
			QEHCParameters* params = new QEHCParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			params->SearchSubject = QEHCParameters::SearchSubjectOption::MinimumContribution;
			params->sort = false;
			params->shuffle = false;
			QEHCResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(result->MinimumContribution > 0.00000237);
			Assert::IsTrue(result->MinimumContribution < 0.00000238);
		}
		TEST_METHOD(FunctionalQEHCmin10)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d6n400_10");
			QEHCSolver solver;
			QEHCParameters* params = new QEHCParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			params->SearchSubject = QEHCParameters::SearchSubjectOption::MinimumContribution;
			params->sort = false;
			params->shuffle = false;
			QEHCResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(result->MinimumContribution > 0.0000031056);
			Assert::IsTrue(result->MinimumContribution < 0.0000031057);
		}
		TEST_METHOD(FunctionalQEHCmin11)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d6n500_10");
			QEHCSolver solver;
			QEHCParameters* params = new QEHCParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			params->SearchSubject = QEHCParameters::SearchSubjectOption::MinimumContribution;
			params->sort = false;
			params->shuffle = false;
			QEHCResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(result->MinimumContribution > 0.00000205);
			Assert::IsTrue(result->MinimumContribution < 0.00000206);
		}
		TEST_METHOD(FunctionalQEHCmin12)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d6n600_10");
			QEHCSolver solver;
			QEHCParameters* params = new QEHCParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			params->SearchSubject = QEHCParameters::SearchSubjectOption::MinimumContribution;
			params->sort = false;
			params->shuffle = false;
			QEHCResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(result->MinimumContribution > 0.000001347);
			Assert::IsTrue(result->MinimumContribution < 0.000001348);
		}
		TEST_METHOD(FunctionalQEHCmin13)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d7n200_10");
			QEHCSolver solver;
			QEHCParameters* params = new QEHCParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			params->SearchSubject = QEHCParameters::SearchSubjectOption::MinimumContribution;
			params->sort = false;
			params->shuffle = false;
			QEHCResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(result->MinimumContribution > 0.0000000000504);
			Assert::IsTrue(result->MinimumContribution < 0.0000000000505);
		}
		TEST_METHOD(FunctionalQEHCmin14)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d7n300_10");
			QEHCSolver solver;
			QEHCParameters* params = new QEHCParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			params->SearchSubject = QEHCParameters::SearchSubjectOption::MinimumContribution;
			params->sort = false;
			params->shuffle = false;
			QEHCResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(result->MinimumContribution > 0.000000001391);
			Assert::IsTrue(result->MinimumContribution < 0.000000001392);
		}
		TEST_METHOD(FunctionalQEHCmin15)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d7n400_10");
			QEHCSolver solver;
			QEHCParameters* params = new QEHCParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			params->SearchSubject = QEHCParameters::SearchSubjectOption::MinimumContribution;
			params->sort = false;
			params->shuffle = false;
			QEHCResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(result->MinimumContribution > 0.00000000009358);
			Assert::IsTrue(result->MinimumContribution < 0.00000000009359);
		}
		TEST_METHOD(FunctionalQEHCmin16)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d7n500_10");
			QEHCSolver solver;
			QEHCParameters* params = new QEHCParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			params->SearchSubject = QEHCParameters::SearchSubjectOption::MinimumContribution;
			params->sort = false;
			params->shuffle = false;
			QEHCResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(result->MinimumContribution > 0.0000000510457);
			Assert::IsTrue(result->MinimumContribution < 0.0000000510458);
		}
	};
	TEST_CLASS(HVETests)
	{
		TEST_METHOD(FunctionalHVEmin1)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n100_10");
			//ds->normalize();
			DBHVESolver solver;
			DBHVEParameters* params = new DBHVEParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			params->MCiterations = 100000;
			DBHVEResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(result->HyperVolumeEstimation > 0.822);
			Assert::IsTrue(result->HyperVolumeEstimation < 0.823);
		}
		TEST_METHOD(FunctionalHVEmin2)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n200_10");
			DBHVESolver solver;
			DBHVEParameters* params = new DBHVEParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			params->MCiterations = 100000;
			DBHVEResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(result->HyperVolumeEstimation > 0.851);
			Assert::IsTrue(result->HyperVolumeEstimation < 0.852);
		}
		TEST_METHOD(FunctionalHVEmin3)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n300_10");
			DBHVESolver solver;
			DBHVEParameters* params = new DBHVEParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			params->MCiterations = 100000;
			params->seed = 22;
			DBHVEResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(result->HyperVolumeEstimation > 0.8803);
			Assert::IsTrue(result->HyperVolumeEstimation < 0.8804);
		}
		TEST_METHOD(FunctionalHVEmin4)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n400_10");
			DBHVESolver solver;
			DBHVEParameters* params = new DBHVEParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			params->MCiterations = 100000;
			params->seed = 22;
			DBHVEResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(result->HyperVolumeEstimation > 0.8790);
			Assert::IsTrue(result->HyperVolumeEstimation < 0.8791);
		}
		TEST_METHOD(FunctionalHVEmin5)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n500_10");
			DBHVESolver solver;
			DBHVEParameters* params = new DBHVEParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			params->MCiterations = 100000;
			params->seed = 22;
			DBHVEResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(result->HyperVolumeEstimation > 0.8880);
			Assert::IsTrue(result->HyperVolumeEstimation < 0.8881);
		}
		TEST_METHOD(FunctionalHVEmin6)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d4n600_10");
			DBHVESolver solver;
			DBHVEParameters* params = new DBHVEParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			params->MCiterations = 100000;
			params->seed = 22;
			DBHVEResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(result->HyperVolumeEstimation > 0.8981);
			Assert::IsTrue(result->HyperVolumeEstimation < 0.8982);
		}
		TEST_METHOD(FunctionalHVEmin7)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d6n100_10");
			DBHVESolver solver;
			DBHVEParameters* params = new DBHVEParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			params->MCiterations = 100000;
			params->seed = 22;
			DBHVEResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(result->HyperVolumeEstimation > 0.46720);
			Assert::IsTrue(result->HyperVolumeEstimation < 0.46722);
		}
		TEST_METHOD(FunctionalHVEmin8)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d6n200_10");
			DBHVESolver solver;
			DBHVEParameters* params = new DBHVEParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			params->MCiterations = 100000;
			params->seed = 22;
			DBHVEResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(result->HyperVolumeEstimation > 0.5354);
			Assert::IsTrue(result->HyperVolumeEstimation < 0.5355);
		}
		TEST_METHOD(FunctionalHVEmin9)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d6n300_10");
			DBHVESolver solver;
			DBHVEParameters* params = new DBHVEParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			params->MCiterations = 100000;
			params->seed = 22;
			DBHVEResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(result->HyperVolumeEstimation > 0.5629);
			Assert::IsTrue(result->HyperVolumeEstimation < 0.5630);
		}
		TEST_METHOD(FunctionalHVEmin10)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d6n400_10");
			DBHVESolver solver;
			DBHVEParameters* params = new DBHVEParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			params->MCiterations = 100000;
			params->seed = 22;
			DBHVEResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(result->HyperVolumeEstimation > 0.6054);
			Assert::IsTrue(result->HyperVolumeEstimation < 0.6055);
		}
		TEST_METHOD(FunctionalHVEmin11)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d6n500_10");
			DBHVESolver solver;
			DBHVEParameters* params = new DBHVEParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			params->MCiterations = 100000;
			params->seed = 22;
			DBHVEResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(result->HyperVolumeEstimation > 0.6311);
			Assert::IsTrue(result->HyperVolumeEstimation < 0.6312);
		}
		TEST_METHOD(FunctionalHVEmin12)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d6n600_10");
			DBHVESolver solver;
			DBHVEParameters* params = new DBHVEParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			params->MCiterations = 100000;
			params->seed = 22;
			DBHVEResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(result->HyperVolumeEstimation > 0.6428);
			Assert::IsTrue(result->HyperVolumeEstimation < 0.6429);
		}
		TEST_METHOD(FunctionalHVEmin13)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d7n200_10");
			DBHVESolver solver;
			DBHVEParameters* params = new DBHVEParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			params->MCiterations = 100000;
			params->seed = 22;
			DBHVEResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(result->HyperVolumeEstimation > 0.005770);
			Assert::IsTrue(result->HyperVolumeEstimation < 0.005780);
		}
		TEST_METHOD(FunctionalHVEmin14)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d7n300_10");
			DBHVESolver solver;
			DBHVEParameters* params = new DBHVEParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			params->MCiterations = 100000;
			params->seed = 22;
			DBHVEResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(result->HyperVolumeEstimation > 0.006560);
			Assert::IsTrue(result->HyperVolumeEstimation < 0.006570);
		}
		TEST_METHOD(FunctionalHVEmin15)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d7n400_10");
			DBHVESolver solver;
			DBHVEParameters* params = new DBHVEParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			params->MCiterations = 100000;
			params->seed = 22;
			DBHVEResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(result->HyperVolumeEstimation > 0.007282);
			Assert::IsTrue(result->HyperVolumeEstimation < 0.007283);
		}
		TEST_METHOD(FunctionalHVEmin16)
		{
			auto ds = DataSet::LoadFromFilename("../../ModaAutomatedTests/Datasets/unit_tests_d7n500_10");
			DBHVESolver solver;
			DBHVEParameters* params = new DBHVEParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			params->MCiterations = 100000;
			params->seed = 22;
			DBHVEResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(result->HyperVolumeEstimation > 0.9500);
			Assert::IsTrue(result->HyperVolumeEstimation < 0.9501);
		}


	};
}

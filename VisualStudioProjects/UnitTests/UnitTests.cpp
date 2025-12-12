#include "pch.h"
#include "CppUnitTest.h"
#include "DataSet.h"
#include "IQHVSolver.h"
#include "DBHVESolver.h"
#include "SolverParameters.h"
#include "QEHCSolver.h"
#include "DBHVESolver.h"
#include "QHV_BQSolver.h"
#include "MCHVESolver.h"

#define TEST_EPSILON 0.002
using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace moda;

namespace UnitTests
{
	TEST_CLASS(UnitTests)
	{
	public:
		// unit tests
		TEST_METHOD(UnitIQHVTest1)
		{
			DataSet* ds = new DataSet(3);
			Point* p1 = new Point(3);
			Point* p2 = new Point(3);
			Point* p3 = new Point(3);

			p1->ObjectiveValues[0] = 0.2;
			p1->ObjectiveValues[1] = 0.2;
			p1->ObjectiveValues[2] = 0.2;

			p2->ObjectiveValues[0] = 0.2;
			p2->ObjectiveValues[1] = 0.2;
			p2->ObjectiveValues[2] = 0.4;

			p3->ObjectiveValues[0] = 0.4;
			p3->ObjectiveValues[1] = 0.2;
			p3->ObjectiveValues[2] = 0.2;

			ds->add(p1);
			ds->add(p2);

			ds->typeOfOptimization = DataSet::OptimizationType::maximization;

			IQHVSolver solver;
			IQHVParameters* params = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			HypervolumeResult* result = solver.Solve(ds, *params);
			Assert::IsTrue(result->HyperVolume > 0.0159);
			Assert::IsTrue(result->HyperVolume < 0.0161);

			ds->add(p3);
			result = solver.Solve(ds, *params);
			Assert::IsTrue(result->HyperVolume > 0.0239);
			Assert::IsTrue(result->HyperVolume < 0.0241);

		}

		TEST_METHOD(UnitNDTree1)
		{
			DataSet* ds = new DataSet(3);
			Point* p1 = new Point(3);
			Point* p2 = new Point(3);
			Point* p3 = new Point(3);

			p1->ObjectiveValues[0] = 0.2;
			p1->ObjectiveValues[1] = 0.4;
			p1->ObjectiveValues[2] = 0.2;

			p2->ObjectiveValues[0] = 0.2;
			p2->ObjectiveValues[1] = 0.2;
			p2->ObjectiveValues[2] = 0.4;

			p3->ObjectiveValues[0] = 0.4;
			p3->ObjectiveValues[1] = 0.2;
			p3->ObjectiveValues[2] = 0.2;

			ds->add(p1);
			ds->add(p2);
			ds->add(p3);
			NDTree<Point> NDTree_ = NDTree<Point>(*ds);
			DataSet* copy = NDTree_.toDataSet();
			Assert::IsTrue(copy->points.size() == 3);
		}

		TEST_METHOD(UnitNDTree2)
		{
			DataSet* ds = new DataSet(3);
			Point* p1 = new Point(3);
			Point* p2 = new Point(3);
			Point* p3 = new Point(3);

			p1->ObjectiveValues[0] = 0.2;
			p1->ObjectiveValues[1] = 0.2;
			p1->ObjectiveValues[2] = 0.2;

			p2->ObjectiveValues[0] = 0.2;
			p2->ObjectiveValues[1] = 0.2;
			p2->ObjectiveValues[2] = 0.4;

			p3->ObjectiveValues[0] = 0.4;
			p3->ObjectiveValues[1] = 0.2;
			p3->ObjectiveValues[2] = 0.2;

			ds->add(p1);
			ds->add(p2);
			ds->add(p3);
			NDTree<Point> NDTree_ = NDTree<Point>(*ds);
			DataSet* copy = NDTree_.toDataSet();

			Assert::IsTrue(copy->points.size() == 2);

		}

		TEST_METHOD(UnitNDTree3)
		{
			DataSet* ds = new DataSet(3);
			Point* p1 = new Point(3);
			Point* p2 = new Point(3);
			Point* p3 = new Point(3);

			p1->ObjectiveValues[0] = 0.1;
			p1->ObjectiveValues[1] = 0.1;
			p1->ObjectiveValues[2] = 0.1;

			p2->ObjectiveValues[0] = 0.2;
			p2->ObjectiveValues[1] = 0.2;
			p2->ObjectiveValues[2] = 0.4;

			p3->ObjectiveValues[0] = 0.4;
			p3->ObjectiveValues[1] = 0.2;
			p3->ObjectiveValues[2] = 0.2;

			ds->add(p1);
			ds->add(p2);
			ds->add(p3);
			NDTree<Point> NDTree_ = NDTree<Point>(*ds);
			DataSet* copy = NDTree_.toDataSet();

			Assert::IsTrue(copy->points.size() == 2);


		}

		TEST_METHOD(UnitNDTree4)
		{
			DataSet* ds = new DataSet(3);
			Point* p1 = new Point(3);
			Point* p2 = new Point(3);
			Point* p3 = new Point(3);

			p1->ObjectiveValues[0] = 0.2;
			p1->ObjectiveValues[1] = 0.2;
			p1->ObjectiveValues[2] = 0.2;

			p2->ObjectiveValues[0] = 0.3;
			p2->ObjectiveValues[1] = 0.3;
			p2->ObjectiveValues[2] = 0.3;

			p3->ObjectiveValues[0] = 0.4;
			p3->ObjectiveValues[1] = 0.4;
			p3->ObjectiveValues[2] = 0.4;

			ds->add(p1);
			ds->add(p2);
			ds->add(p3);
			NDTree<Point> NDTree_ = NDTree<Point>(*ds);
			Assert::IsTrue(NDTree_.numberOfSolutions() == 1);
			DataSet* copy = NDTree_.toDataSet();

			//Assert::IsTrue(copy->points.size() == 1);


		}
		TEST_METHOD(UnitNDTree5_2) {

			//wygeneruj du¿y zbiór punktów
				// generowanie punktów powinno odbywaæ siê w dok³adnie taki sam sposób jak w HVE (+randomizowany promieñ
				// dziedzina wartoœci powinna byæ stosunkowo ma³a (np 10 ró¿nych wartoœci)
				// liczba sk³adowych powinna siê ró¿niæ w ró¿nych testach (2,4,6,7,8,9)
				// liczba punktów powinna byæ równa 1000

			int numberOfObjectives = 2;
			DType radius = 1.0;
			DType variance = 0;
			std::uniform_real_distribution<DType> uniformDType(0, 1);
			std::normal_distribution<DType> normal(0, 1);
			std::default_random_engine randEng;
			DataSet* ds = new DataSet(numberOfObjectives);
			randEng.seed(123);
			for (int i = 0; i < 1000; i++)
			{
				std::vector <DType> randomDirectionVector;
				randomDirectionVector.resize(numberOfObjectives);

				short j;
				for (j = 0; j < numberOfObjectives; j++) {
					randomDirectionVector[j] = normal(randEng);
					if (randomDirectionVector[j] < 0) {
						randomDirectionVector[j] = -randomDirectionVector[j];
					}
					if (randomDirectionVector[j] == 0) {
						randomDirectionVector[j] = 0.00000000001;
					}
				}
				Normalize(randomDirectionVector, numberOfObjectives, radius, variance);
				Point* p = new Point(numberOfObjectives);
				std::copy(randomDirectionVector.begin(), randomDirectionVector.end(), p->ObjectiveValues);
				ds->add(p);
			}


			//przekonweruj pierwsz¹ kopiê na NDTree
			NDTree<Point> NDTree_ = NDTree<Point>(*ds);
			DataSet* NDTreeFlat = NDTree_.toDataSet();
			//przekonweruj drug¹ kopiê na ListSet
			ListSet<Point> LSet = ListSet<Point>(*ds);
			//dla ka¿dego punktu w listsecie
			for (Point* p1 : LSet)
			{
				bool contains = true;
				//sprwadŸ czy jest w drzewie, je¿eli nie to zakoñcz test niepowodzeniem
				for (Point* p2 : NDTreeFlat->points)
				{
					bool equal = true;
					for (int objective = 0; objective < p2->NumberOfObjectives; objective++)
					{
						if (p1->ObjectiveValues[objective] != p2->ObjectiveValues[objective])
						{
							equal = false;
							break;
						}
					}
					contains = equal;
					if (contains == false)
						break;
				}
				Assert::AreEqual(contains, true);
			}
			//dla ka¿dego punktu w NDTree
		}
		void Normalize(std::vector <DType>& p, int numberOfObjectives, DType radius, DType variance) {
			DType randomized_radius = radius + ((DType)rand() / (DType)RAND_MAX - 0.5) * radius * variance;
			DType nrm = Norm(p, numberOfObjectives, randomized_radius);
			DType s = 0;
			int j;
			for (j = 0; j < numberOfObjectives; j++) {
				p[j] /= nrm;
			}
		}
		DType Norm(std::vector <DType>& p, int numberOfObjectives, DType radius) {
			DType s = 0;
			int j;
			for (j = 0; j < numberOfObjectives; j++) {
				s += p[j] * p[j];
			}
			return sqrt(s) / radius;
		}
		TEST_METHOD(UnitTestBQHV_IQHV)
		{
			int numberOfObjectives = 3;
			DType radius = 1.0;
			DType variance = 0.1;
			int wrong = 0;
			int no_wrongs = 5;
			std::uniform_real_distribution<DType> uniformDType(0, 1);
			std::normal_distribution<DType> normal(0, 1);
			std::default_random_engine randEng;
			for (int test_no = 0; test_no < 100; test_no++)
			{
				DataSet* ds = new DataSet(numberOfObjectives);
				randEng.seed(123);
				for (int i = 0; i < 1000; i++)
				{
					std::vector <DType> randomDirectionVector;
					randomDirectionVector.resize(numberOfObjectives);
					randEng.seed(i + 123);
					short j;
					for (j = 0; j < numberOfObjectives; j++) {
						randomDirectionVector[j] = normal(randEng);
						if (randomDirectionVector[j] < 0) {
							randomDirectionVector[j] = -randomDirectionVector[j];
						}
						if (randomDirectionVector[j] == 0) {
							randomDirectionVector[j] = 0.00000000001;
						}
					}
					Normalize(randomDirectionVector, numberOfObjectives, radius, variance);
					Point* p = new Point(numberOfObjectives);
					std::copy(randomDirectionVector.begin(), randomDirectionVector.end(), p->ObjectiveValues);
					ds->add(p);
				}
				//oblicz wartoœci lower i upper bound dla tego datasetu
				//oblicz wartoœæ hypervolume dla tego datasetu

				//wylosuj dataset
				IQHVSolver iqhvSolver;
				IQHVParameters iqhvParams;
				auto hv_result = iqhvSolver.Solve(ds, iqhvParams);

				QHV_BQSolver bqSolver;
				QHV_BQParameters bqParams;
				auto bounded_result = bqSolver.Solve(ds, bqParams);
				//sprawdŸ, czy hypervolume jest pomiêdzy lb i ub
				if (bounded_result->LowerBound > hv_result->HyperVolume || bounded_result->UpperBound < hv_result->HyperVolume)
					wrong++;
			}
			//wykonaj powy¿sze kroki 100 razy
			Assert::IsTrue(wrong < no_wrongs);
		}

		//QEHCSolver
		//liczê contributions brute force
		//sprawdzam czy qehc daje to samo min/max


		TEST_METHOD(UnitTestMCHV_IQHV)
		{
			int wrong = 0;
			int numberOfObjectives = 3;
			DType radius = 1.0;
			DType variance = 0.1;
			int no_wrongs = 1;
			std::uniform_real_distribution<DType> uniformDType(0, 1);
			std::normal_distribution<DType> normal(0, 1);
			std::default_random_engine randEng;
			for (int test_no = 0; test_no < 100; test_no++)
			{
				DataSet* ds = new DataSet(numberOfObjectives);
				randEng.seed(123 + test_no);
				for (int i = 0; i < 1000; i++)
				{
					std::vector <DType> randomDirectionVector;
					randomDirectionVector.resize(numberOfObjectives);

					short j;
					for (j = 0; j < numberOfObjectives; j++) {
						randomDirectionVector[j] = normal(randEng);
						if (randomDirectionVector[j] < 0) {
							randomDirectionVector[j] = -randomDirectionVector[j];
						}
						if (randomDirectionVector[j] == 0) {
							randomDirectionVector[j] = 0.00000000001;
						}
					}
					Normalize(randomDirectionVector, numberOfObjectives, radius, variance);
					Point* p = new Point(numberOfObjectives);
					std::copy(randomDirectionVector.begin(), randomDirectionVector.end(), p->ObjectiveValues);
					ds->add(p);
				}
				//oblicz wartoœci lower i upper bound dla tego datasetu
				//oblicz wartoœæ hypervolume dla tego datasetu

				//wylosuj dataset
				IQHVSolver iqhvSolver;
				IQHVParameters iqhvParams;
				auto hv_result = iqhvSolver.Solve(ds, iqhvParams);

				MCHVESolver mcSolver;
				MCHVParameters bqParams;
				bqParams.MaxEstimationTime = 100;
				auto bounded_result = mcSolver.Solve(ds, bqParams);
				//sprawdŸ, czy hypervolume jest pomiêdzy lb i ub
				if (bounded_result->LowerBound > hv_result->HyperVolume || bounded_result->UpperBound < hv_result->HyperVolume)
					wrong++;
				ds->clear();

			}
			Assert::IsTrue(wrong < no_wrongs);
		}
		TEST_METHOD(UnitIQHVandHVETest1)
		{
			DataSet* ds = new DataSet(3);
			Point* p1 = new Point(3);
			Point* p2 = new Point(3);
			Point* p3 = new Point(3);

			p1->ObjectiveValues[0] = 0.2;
			p1->ObjectiveValues[1] = 0.2;
			p1->ObjectiveValues[2] = 0.2;

			p2->ObjectiveValues[0] = 0.2;
			p2->ObjectiveValues[1] = 0.2;
			p2->ObjectiveValues[2] = 0.4;

			p3->ObjectiveValues[0] = 0.4;
			p3->ObjectiveValues[1] = 0.2;
			p3->ObjectiveValues[2] = 0.2;

			ds->add(p1);
			ds->add(p2);
			ds->add(p3);

			ds->typeOfOptimization = DataSet::OptimizationType::maximization;

			IQHVSolver solver;
			IQHVParameters* params = new IQHVParameters(SolverParameters::zeroone, SolverParameters::zeroone);
			DBHVESolver hveSolver;
			DBHVEParameters* hveParams = new DBHVEParameters(SolverParameters::zeroone, SolverParameters::zeroone);

			HypervolumeResult* result = solver.Solve(ds, *params);
			BoundedResult* boundedResult = hveSolver.Solve(ds, *hveParams);

			Assert::IsTrue(result->HyperVolume > 0.0239);
			Assert::IsTrue(result->HyperVolume < 0.0241);
			Assert::IsTrue(result->HyperVolume >= boundedResult->HyperVolumeEstimation - boundedResult->HyperVolumeEstimation * 0.1);
			Assert::IsTrue(result->HyperVolume <= boundedResult->HyperVolumeEstimation + boundedResult->HyperVolumeEstimation * 0.1);

		}
	};
}

#include "DBHVE.h"
#include "MemoryManager.h"
namespace moda {
    namespace backend {
        DType DBHVE(int memorySlot, Point& idealPoint, Point& nadirPoint, int numberOfSolutions, int MCiterations, clock_t runtimeLimit,
            unsigned seed, int numberOfObjectives, Point& referencePoint, clock_t& it0)
        {

#if UNDERLYING_TYPE == 1
            myvector<Point*>* set;
#elif UNDERLYING_TYPE == 2
            SemiDynamicArray<Point*>* set;
#elif UNDERLYING_TYPE == 3
            SecureVector<Point*>* set;
#else
            std::vector<Point*>* set;
#endif
            int maxIndexUsed = ContextPool::getInstance().maxIndexUsedNumbers[memorySlot];

            set = ContextPool::getInstance().PointsTable[memorySlot];
            std::uniform_real_distribution<DType> uniformDType(0, 1);
            std::normal_distribution<DType> normal(0, 1);
            std::default_random_engine randEng;
            if (seed > 0) {
                randEng.seed(seed);
            }

            std::vector <DType> unitPoint;
            unitPoint.resize(numberOfObjectives, 1);

            std::vector <DType> randomDirectionVector;
            randomDirectionVector.resize(numberOfObjectives);

            std::vector <DType> randomWeightsVector;
            randomWeightsVector.resize(numberOfObjectives);

            referencePoint = idealPoint;

            // Put all points in ND-Tree
            NDTree<Point> NDTree_;
            int ii;
            for (ii = 0; ii < numberOfSolutions; ii++) {
                NDTree_.update(*set->at(ii), false);
                //if (ii < 300 && ii % 100 == 0)
                //{
                //    NDTree_.root->print(0);
                //    std::cout << "-------------------------\n";
                //}
            }

            DType sumDominating = 0;
            DType factor = pow(PI, numberOfObjectives / 2.0) /
                tgamma(numberOfObjectives / 2.0 + 1) *
                pow(0.5, numberOfObjectives);
            unsigned elapsedIterations;
            for (elapsedIterations = 0; elapsedIterations < MCiterations; elapsedIterations++) {
                if (clock() - it0 >= runtimeLimit) {
                    //                break;
                }
                // Draw a random direction and corresponding weight vector

                // Draw a random point (drection) on the surface of a ball with radius 1
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
                Backend::Normalize(randomDirectionVector, numberOfObjectives);

                for (j = 0; j < numberOfObjectives; j++) {

                    randomWeightsVector[j] = -1 / randomDirectionVector[j];
                }
/*                DType maxS = -1e30;
                int maxIndex = -1;
                int ii;
                for (ii = 0; ii < numberOfSolutions; ii++) {
                    DType s = (*set)[ii]->CleanChebycheffScalarizingFunctionInverse(randomWeightsVector, nadirPoint);
                    if (maxS < s) {
                        maxS = s;
                        maxIndex = ii;
                    }
                }
                sumDominating += pow(maxS, numberOfObjectives); */           
                BestSolution bestSolution;
                DType currentMax = -1e30;
                NDTree_.root->maxScalarizingFunction(currentMax, nadirPoint, randomWeightsVector, bestSolution);


                // Calculate Euclidean distance to this point
                sumDominating += pow(currentMax, numberOfObjectives);


                DType estimation = sumDominating * factor / elapsedIterations;
                //BoundedResult* r = new BoundedResult;
                //r->type = Result::ResultType::Estimation;
                //r->HyperVolumeEstimation = estimation;

                //r->LowerBound = estimation;
                //r->UpperBound = estimation;
                //r->ElapsedTime = clock() - it0;
                //if (MCiterations == UINT_MAX)
                //{
                //    IterationCallback(r->ElapsedTime / 10, runtimeLimit / 10, r);
                //}
                //else
                //{
                //    IterationCallback(elapsedIterations, MCiterations, r);
                //}


            }
            
            DType sumDominatingFactor = pow(PI, numberOfObjectives / 2.0) / tgamma(numberOfObjectives / 2.0 + 1.0) * pow(0.5, numberOfObjectives) /
                (DType)elapsedIterations;
            //BoundedResult* r = new BoundedResult;
            //r->type = Result::ResultType::Estimation;
            //r->HyperVolumeEstimation = sumDominating * sumDominatingFactor;
            //r->LowerBound = sumDominating * sumDominatingFactor;
            //r->UpperBound = sumDominating * sumDominatingFactor;
            //r->ElapsedTime = clock() - it0;
            //if (MCiterations == UINT_MAX)
            //{
            //    IterationCallback(r->ElapsedTime / 10, runtimeLimit / 10, r);
            //}
            //else
            //{
            //    IterationCallback(elapsedIterations, MCiterations, r);
            //}

            return sumDominating * sumDominatingFactor;
        }
    }
}
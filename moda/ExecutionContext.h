#pragma once
#include "Point.h"
#include "DynamicStructures.h"
#include "ProcessData.h"
#include "SubproblemsPool.h"
#include "SubProblem.h"
#define UNDERLYING_TYPE 4
namespace moda {
    namespace backend {
        class ExecutionContext
        {
        public:
            enum ExecutionContextType {
                IQHVContext,
                QEHCContext
            };

            ExecutionContextType type;
            explicit ExecutionContext(int reserveSize, int initialSize, int numberOfObjectives, bool shallow = false);
#if UNDERLYING_TYPE == 1
            myvector<Point*>* points;
#elif  UNDERLYING_TYPE == 2
            SemiDynamicArray<Point*>* points;
#elif UNDERLYING_TYPE == 3
            SecureVector<Point*>* points;
#elif UNDERLYING_TYPE == 4
            std::vector<Point*>* points;
#endif

            double result = 0.0;
            bool shallow = false;
            int initialSize = 0;
            int numObjectives = 0;


            std::vector<int> objectivesOrder;
            virtual ~ExecutionContext() {
                
                    // Check if we own the data before cleaning up the contents
                    if (points != nullptr && !shallow) {
                        // We need a way to clean up the Point* objects managed by the array.
                        // Assuming you add a function like `deleteContainedObjects` to SemiDynamicArray:
                        //points->deleteContainedPoints(initialSize); // <-- Cleans up Point objects AND NULLED THEM
                    }
                    delete points;
                    // Always delete the SemiDynamicArray object itself, as the Context owns the container pointer.
                    //delete points;
                
            }

        };

        class IQHVExecutionContext : public ExecutionContext
        {
        public:
            ~IQHVExecutionContext() {
                //delete idealPoint;
                //delete nadirPoint;
                //delete points;
                objectivesOrder.clear();
            }
            explicit IQHVExecutionContext(int reserveSize, int initialSize, int numberOfObjectives, bool shallow = false);
            int maxIndexUsed = 0;
            int maxIndexUsedOverall = 0;
            Point* idealPoint = nullptr;
            Point* nadirPoint = nullptr;
        };
        class QEHCExecutionContext : public ExecutionContext
        {
        public:
            ~QEHCExecutionContext() {
                //ownership problems
                objectivesOrder.clear();
            }
            explicit QEHCExecutionContext(int reserveSize, int initialSize, int numberOfObjectives, bool shallow = false);
            int maxIndexUsed = 0;
            std::shared_ptr<ProcessData> process;
            std::shared_ptr<SubproblemsPool<SubProblem>> subProblemsPool;
            DType maxContributionLowerBound = 0;
            int lowerBoundProcessId = -1;
            DType minContributionUpperBound = 1;
            int upperBoundProcessId = -1;

            int iPos = 0;
        };
    }
}
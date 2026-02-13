#pragma once
#include "ExecutionContext.h"
namespace moda {
    namespace backend {

        ExecutionContext::ExecutionContext(int reserveSize, int initialSize, int numberOfObjectives, bool shallow) {
#if UNDERLYING_TYPE == 1
            points = new myvector<Point*>();
#elif UNDERLYING_TYPE == 2
            points = new SemiDynamicArray<Point*>();
#elif UNDERLYING_TYPE == 3
            points = new SecureVector<Point*>();
#else
            points = new std::vector<Point*>();
#endif
            points->reserve(reserveSize);
            points->resize(reserveSize);
            this->initialSize = initialSize;
            this->shallow = shallow;
            this->numObjectives = numberOfObjectives;
            for(int i = 0; i<numberOfObjectives; i++) this->objectivesOrder.push_back(i);
            
        }
        IQHVExecutionContext::IQHVExecutionContext(int reserveSize, int initialSize, int numberOfObjectives, bool shallow) : ExecutionContext(reserveSize, initialSize, numberOfObjectives, shallow)
        {
            type = ExecutionContext::ExecutionContextType::IQHVContext;
        }

        QEHCExecutionContext::QEHCExecutionContext(int reserveSize, int initialSize, int numberOfObjectives, bool shallow) : ExecutionContext(reserveSize, initialSize, numberOfObjectives, shallow)
        {
            type = ExecutionContext::ExecutionContextType::QEHCContext;
            
        }
    }
}
#pragma once
#include "ExecutionPool.h"
namespace moda {
    namespace backend {
        int ExecutionPool::reserveContext(int reserveSize, int initialSize, int numberOfObjectives, ExecutionContext::ExecutionContextType contextType, bool shallow) {
			ExecutionPool::mutex.lock();
			int slot = -1;

			for (int i = 0; i < MEMORY_POOL_SIZE; i++)
			{
				if (std::find(reservedIds.begin(), reservedIds.end(), i) == reservedIds.end())
				{
					slot = i;
					break;
				}
			}
			if (slot >= 0)
			{
				ExecutionContext* context = nullptr;
				switch (contextType)
				{
				case moda::backend::ExecutionContext::IQHVContext:
					context = new IQHVExecutionContext(reserveSize, initialSize, numberOfObjectives, shallow);
					break;
				case moda::backend::ExecutionContext::QEHCContext:
					context = new QEHCExecutionContext(reserveSize, initialSize, numberOfObjectives, shallow);
					break;
				default:
					break;
				}
				
				contexts[slot] = context;
				reservedIds.push_back(slot);
			}
			ExecutionPool::mutex.unlock();

			return slot;
        }
		void ExecutionPool::releaseContext(int slot) {
			ExecutionPool::mutex.lock();
			auto context = contexts[slot];


			if (auto iqhv = dynamic_cast<IQHVExecutionContext*>(contexts[slot])) {
				delete iqhv->nadirPoint;
				delete iqhv->idealPoint;
				for (int i = 0; i < context->initialSize; i++)
				{
					
#if UNDERLYING_TYPE == 1
						Point* point = context->points->at(i);
#elif UNDERLYING_TYPE == 2

						Point* point = context->points->at(i);
#elif UNDERLYING_TYPE == 3
						Point* point = context->points->at(i);
#else
						Point* point = context->points->at(i);
#endif
					delete point;
					context->points->at(i) = nullptr;
				}

				context->objectivesOrder.clear();

				if (context->initialSize != 0)
					delete context->points;
			}

			if (auto qehc = dynamic_cast<QEHCExecutionContext*>(contexts[slot])) //poly/inheritance
			//if(contexts[slot]->type == ExecutionContext::ExecutionContextType::QEHCContext) //typing
			{
				QEHCExecutionContext* masked = (QEHCExecutionContext*)contexts[slot];
				delete masked->process;
				delete context->points;
				
			}
			reservedIds.erase(std::remove(reservedIds.begin(), reservedIds.end(), slot), reservedIds.end());
			delete contexts[slot];
			ExecutionPool::mutex.unlock();
		}
		bool ExecutionPool::full()
		{
			for (int i = 0; i < MEMORY_POOL_SIZE; i++)
			{
				if (std::find(reservedIds.begin(), reservedIds.end(), i) == reservedIds.end())
				{
					return false;
				}
			}
			return true;
		}
		ExecutionContext* ExecutionPool::getContext(int id)
		{

			//fast and not really safe - is this an issue?
			return contexts[id];
			//slow and safe
			//ExecutionPool::mutex.lock();
			//if (std::find(reservedIds.begin(), reservedIds.end(), id) != reservedIds.end())
			//{
			//	ExecutionPool::mutex.unlock();
			//	return contexts[id];
			//}
			//ExecutionPool::mutex.unlock();
			//return nullptr;

		}
		ExecutionPool::~ExecutionPool()
		{
			//for (int i = 0; i < MEMORY_POOL_SIZE; i++)
			//{
			//	if(contexts[i] != nullptr)
			//		delete contexts[i];
			//}
		}
		ExecutionPool::ExecutionPool()
		{
			//ExecutionPool::mutex.lock();
			//for (int i = 0; i < MEMORY_POOL_SIZE; i++)
			//{
			//	contexts[i] = new ExecutionContext(0, 0, 0);
			//}
			//ExecutionPool::mutex.unlock();
		}

    }
}
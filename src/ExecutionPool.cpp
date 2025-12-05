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
			delete context;
			reservedIds.erase(std::remove(reservedIds.begin(), reservedIds.end(), slot), reservedIds.end());
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
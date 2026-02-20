#include "MemoryManager.h"

namespace moda
{
	namespace backend {
		std::mutex ContextPool::accessible;
		std::random_device ContextPool::randomDevice;
		std::mt19937 ContextPool::randomNumberGenerator(randomDevice());
		bool ContextPool::full()
		{
			for (int i = 0; i < MEMORY_POOL_SIZE; i++)
			{
				if (std::find(takenSlots.begin(), takenSlots.end(), i) == takenSlots.end())
				{
					return false;
				}
			}
			return true;
		}

		ContextPool::ContextPool() {
			ContextPool::accessible.lock();
			for(int i = 0; i< MEMORY_POOL_SIZE; i++)
#if UNDERLYING_TYPE == 1
				PointsTable[i] = new myvector<Point*>();
#elif UNDERLYING_TYPE == 2
				PointsTable[i] = new SemiDynamicArray<Point*>();
#elif UNDERLYING_TYPE == 3
				PointsTable[i] = new SecureVector<Point*>();
#else

				PointsTable[i] = new std::vector<Point*>();
#endif
			ContextPool::accessible.unlock();
		};
		void ContextPool::releaseMemory(int slot)
		{
			ContextPool::accessible.lock();
			
			for (int i = 0; i < initialSizes[slot]; i++)
			{
#if UNDERLYING_TYPE == 1
				Point* point = PointsTable[slot]->at(i);
#elif UNDERLYING_TYPE == 2
				Point* point = PointsTable[slot]->at(i);
#else
				Point* point = PointsTable[slot]->at(i);
#endif
				delete point;
			}
			PointsTable[slot]->clear();
			PointsTable[slot]->resize(0);
			//if(initialSizes[slot] != 0)
				delete PointsTable[slot];

#if UNDERLYING_TYPE == 1
			PointsTable[slot] = new myvector<Point*>();
#elif UNDERLYING_TYPE == 2
			PointsTable[slot] = new SemiDynamicArray<Point*>();
#elif UNDERLYING_TYPE == 3
			PointsTable[slot] = new SecureVector<Point*>();

#else
			PointsTable[slot] = new std::vector<Point*>();
#endif
			initialSizes[slot] = 0;
			takenSlots.erase(std::remove(takenSlots.begin(), takenSlots.end(), slot), takenSlots.end());
			ContextPool::accessible.unlock();
		}
		int ContextPool::reservePoints(int no_points, int initial_size, int numberOfObjectives, bool reference_memory)
		{

			ContextPool::accessible.lock();
			int slot = -1;
			
			for (int i = 0; i < MEMORY_POOL_SIZE; i++)
			{
				if (std::find(takenSlots.begin(), takenSlots.end(), i) == takenSlots.end())
				{
					slot = i;
					break;
				}
			}
			if (slot >= 0)
			{
				if (reference_memory) initialSizes[slot] = 0;
				else initialSizes[slot] = initial_size;
				reservedSizes[slot] = initial_size;
				PointsTable[slot]->resize(no_points);
				takenSlots.push_back(slot);
				objectivesOrder[slot].clear();
				for(int i = 0; i<numberOfObjectives; i++)
					objectivesOrder[slot].push_back(i);
				
			}
			ContextPool::accessible.unlock();

			return slot;
		}
		ContextPool& ContextPool::getInstance()
		{
			static ContextPool instance;
			return instance;
		}

		void ContextPool::offsetObjectives(int slot, int offset)
		{
			if(offset > 0)
				std::rotate(objectivesOrder[slot].begin(), objectivesOrder[slot].begin() + offset, objectivesOrder[slot].end());
			if(offset < 0)
				std::rotate(objectivesOrder[slot].rbegin(), objectivesOrder[slot].rbegin() - offset, objectivesOrder[slot].rend());

		}

		void ContextPool::shuffleObjectives(int slot)
		{
			std::shuffle(objectivesOrder[slot].begin(), objectivesOrder[slot].end(), randomNumberGenerator);
		}

		void ContextPool::sortObjectives(int slot, std::vector<int> order)
		{
			std::sort(objectivesOrder[slot].begin(), objectivesOrder[slot].end(), [&](int a, int b) {
				return order[a] < order[b];
				});
		}
	}
}

#pragma once
#include <map>
#include "Point.h"
#include "DynamicStructures.h"
#include <tuple>
#include <mutex>

#define UNDERLYING_TYPE 4
#define MEMORY_POOL_SIZE 1024
namespace moda {
	namespace backend {
		class ContextPool {
		public:
			static std::random_device randomDevice;
			static std::mt19937 randomNumberGenerator;
			/// <summary>
			/// Get memory instance
			/// </summary>
			/// <returns></returns>
			static ContextPool& getInstance();
			/// <summary>
			/// Multi-threading access variable
			/// </summary>
			static std::mutex accessible;
			/// <summary>
			/// Memory reservation sizes
			/// </summary>
			std::vector<std::tuple<int, int>> reservations;
			/// <summary>
			/// Memory table
			/// </summary>

			#if UNDERLYING_TYPE == 1
			myvector<Point*>* PointsTable[MEMORY_POOL_SIZE];
			#elif  UNDERLYING_TYPE == 2
			SemiDynamicArray<Point*>* PointsTable[MEMORY_POOL_SIZE];
			#elif UNDERLYING_TYPE == 3
			SecureVector<Point*>* PointsTable[64];
			#elif UNDERLYING_TYPE == 4		
			std::vector<Point*>* PointsTable[64];
			#endif

			/// <summary>
			/// Ideal points for separate executeions
			/// </summary>
			Point* idealPoints[MEMORY_POOL_SIZE];
			/// <summary>
			/// Nadir points for separate executions
			/// </summary>
			Point* nadirPoints[MEMORY_POOL_SIZE];
			/// <summary>
			/// Execution results
			/// </summary>
			double results[MEMORY_POOL_SIZE];
			/// <summary>
			/// Initial sizes of reserved memory - used for memory release
			/// </summary>
			int initialSizes[MEMORY_POOL_SIZE];
			/// <summary>
			/// Sizes of reserved memory - used for shuffling
			/// </summary>
			int reservedSizes[MEMORY_POOL_SIZE];
			/// <summary>
			/// Current maximum index used
			/// </summary>
			int maxIndexUsedNumbers[MEMORY_POOL_SIZE];
			/// <summary>
			/// Experimental parameter - stores the maximum index used over the course of a single execution
			/// </summary>
			int maxMaxIndexUsedNumbers[MEMORY_POOL_SIZE];
			/// <summary>
			/// Numbers of objectives for problems
			/// </summary>
			int numbersOfObjectives[MEMORY_POOL_SIZE];
			/// <summary>
			/// Reserve memory
			/// </summary>
			/// <param name="no_points">initial memory size</param>
			/// <param name="initial_size">number of points to be initially stored in the memory - used for memory release</param>
			/// <param name="numberOfObjectives">number of objectives</param>
			/// <param name="reference_memory">is it a deep copy of memory values, or a set of references?</param>
			/// <returns>Memory slot. If no memory slots are available, function returns -1. It is recommended to use existing memory slot, or wait for existing execution to end, id that's the case.</returns>
			int reservePoints(int no_points, int initial_size, int numberOfObjectives ,bool reference_memory = false);
			/// <summary>
			/// Release memory
			/// </summary>
			/// <param name="slot">Memory slot. </param>
			void releaseMemory(int slot);
			/// <summary>
			/// Checks if memory pool is full
			/// </summary>
			/// <returns>Indication if there is any space in memory pool</returns>
			bool full();
			std::vector<int> objectivesOrder[MEMORY_POOL_SIZE];

			void offsetObjectives(int slot, int offset);

			void shuffleObjectives(int slot);

			void sortObjectives(int slot, std::vector<int> order);
			void cleanMemory();
		private:
			std::vector<int> takenSlots;
			ContextPool();
			//removing access from dangerous procedures to solve singleton duplication issues
			ContextPool(ContextPool const&);
			void operator=(ContextPool const&);
		};
		
		
	}

}
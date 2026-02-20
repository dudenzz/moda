#include "SubproblemsPool.h"

namespace moda {

	template<class T> SubproblemsPool<T>::SubproblemsPool()
	{
		clock_t t0 = clock();
		this->reserve(10000000); // original value: 1000000000
		this->resize(1); // original value: 10000000
	}
	template<class T> int SubproblemsPool<T>::getNew()
	{
		if (pool.size() > 0) {

			int index = pool.top();
			pool.pop();
			return index;
		}
		else {
			if (maxElementUsed >= this->size() - 1) {
				this->resize(this->size() + 10000000); // !!! 10000000
			}
			return ++maxElementUsed;
		}

	}
	template<class T> void SubproblemsPool<T>::free(int index)
	{
		pool.push(index);
	}

	template<class T> void SubproblemsPool<T>::clear()
	{
		maxElementUsed = -1;
		pool = {};
		myvector<T>::clear();
	}




	template SubproblemsPool<SubProblem>::SubproblemsPool();
	template int SubproblemsPool<SubProblem>::getNew();
	template void SubproblemsPool<SubProblem>::free(int index);
	template void SubproblemsPool<SubProblem>::clear();
}
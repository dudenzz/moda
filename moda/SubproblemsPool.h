#pragma once
#include "SubProblem.h"
#include "myvector.h"
#include "vector"
namespace moda {
	// Class used for subproblems memory management intead of standard new and delete
	template <class T>
	class SubproblemsPool : public myvector<T> {
	private:
		int maxElementUsed = -1;

		std::priority_queue<int> pool;

	public:


		SubproblemsPool();
		int getNew();
		void free(int index);
		void clear();
	};

}
#pragma once

#include "SubProblem.h"


namespace moda {

	template <class T>
	class myvector {
	protected:
		int shift = 14; //22
		int base = (1 << shift);
		int maxvectorsize = 1 << 22;
		int mask = base - 1;
		std::vector <std::vector <T>> vec;

		int row;
		int col;

		int currentMaxRow = 0;

	public:
		myvector();
		void reserve(int newsize);

		int size();

		void resize(int newSize);
		inline T& operator[](const int index) {
			return vec[index >> shift][index & mask];
		}

		inline const T& get(const int index) {
			return vec[index >> shift][index & mask];
		}

		void clear();
	};
}
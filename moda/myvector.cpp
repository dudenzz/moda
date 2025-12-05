#include "myvector.h"

namespace moda {

	template<class T> void myvector<T>::reserve(int newsize) {}
	template<class T> myvector<T>::myvector() {
		vec.resize(base);
	}
	template<class T> myvector<T>::~myvector() {
		vec.clear();
	}
	template<class T> int myvector<T>::size()
	{
		return (currentMaxRow + 1) * base;
	}
	template<class T> void myvector<T>::resize(int newSize)
	{
		if (newSize < size()) {
			currentMaxRow = 0;
		}

		int newRows = 1 + (newSize - size()) / base;
		int i;
		if (newSize / base >= vec.size()) vec.resize(newSize / base * 2);
		for (i = currentMaxRow; i <= currentMaxRow + newRows; i++) {
			vec[i].resize(base);
		}
		currentMaxRow += newRows;
	}
	template<class T> void myvector<T>::clear()
	{
		currentMaxRow = 0;
	}


	template void myvector<Point*>::reserve(int newsize);
	template void myvector<SubProblem>::reserve(int newsize);
	template void myvector<Point*>::resize(int newsize);
	template void myvector<SubProblem>::resize(int newsize);
	template int myvector<Point*>::size();
	template int myvector<SubProblem>::size();
	template void myvector<Point*>::clear();
	template void myvector<SubProblem>::clear();
	template myvector<Point*>::myvector();
	template myvector<SubProblem>::myvector();
	template myvector<Point*>::~myvector();
	template myvector<SubProblem>::~myvector();

}
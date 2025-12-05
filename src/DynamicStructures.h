#pragma once
#include "Point.h"
#include "SubProblem.h"

namespace moda {
    namespace backend {
        template <typename T>
        class SecureVector : public std::vector<T>
        {
        public:
            int capacity = 1;

            void reserve(size_t newCapacity)  {
                capacity = newCapacity;
                std::vector<T>::reserve(newCapacity);
                //std::vector<T>::resize(newCapacity);
            }
            inline T& operator[](size_t index) {
                if (index >= capacity)
                {
                    capacity *= 2;
                    std::vector<T>::resize(index * 2);
                }
                return std::vector<T>::operator[](index);
            }
            inline T& at(size_t index) {
                if (index >= capacity)
                {
                    capacity *= 2;
                    std::vector<T>::resize(index * 2);
                }
                return std::vector<T>::operator[](index);
            }
        };

        template <typename T>
        class SemiDynamicArray {
        public:
            std::unique_ptr<T[]> data;
            size_t size = 0;
            size_t capacity = 0;
            int maxCall;
            SemiDynamicArray() {
                data = std::make_unique<T[]>(2000);
                capacity = 2000;
                size = 0;
                maxCall = 0;
            }

            ~SemiDynamicArray() {

            }

            /**
             * @brief Deletes the objects pointed to by the pointers in the array
             * and sets the array element to nullptr.
             * @note This should ONLY be called by the owning context.
             */
            void deleteContainedPoints(int initialSize) {

                // Iterate up to the deep memory total size, or maxCall if it is smaller.
                int max = maxCall + 1 < initialSize ? maxCall + 1 : initialSize;
                for (int i = 0; i < max; i++) 
                {
                    delete data[i];
                }
                //Reset maxCall/size
                maxCall = 0; 
                size = 0;
            }
            SemiDynamicArray(const SemiDynamicArray&) = delete;
            SemiDynamicArray& operator=(const SemiDynamicArray&) = delete;
            void reserve(size_t newCapacity) {
                if (newCapacity <= capacity)
                    return;

                // 1. Allocate the new, larger array
                // We get a raw pointer to the new memory block.
                T* newDataRaw = new T[newCapacity](); // Use () for zero-initialization of pointers

                // 2. Get the raw pointer to the old memory block
                T* oldDataRaw = data.release(); // unique_ptr gives up ownership and returns the raw pointer. 
                // 'data' is now null. This is crucial.

                // 3. Transfer the contents (Point* addresses)
                for (size_t i = 0; i < size; ++i)
                {
                    // Transfer the pointer value
                    newDataRaw[i] = oldDataRaw[i];

                    // Since oldDataRaw is about to be deleted[], we don't need to null the pointers
                    // within the array unless T has a custom destructor/deleter, 
                    // which Point* does not.
                }

                // 4. Delete the old contiguous array block (the correct way for a dynamic array)
                if (oldDataRaw != nullptr) {
                    delete[] oldDataRaw; // *** GUARANTEED DELETION OF THE OLD BLOCK ***
                }

                // 5. Update the main data pointer (and capacity)
                data.reset(newDataRaw); // unique_ptr takes ownership of the new block
                capacity = newCapacity;
            }

            void resize(size_t newSize) {
                if (newSize > capacity) {
                    // Case 1: Grow AND Reallocate
                    reserve(newSize);
                }

                // Case 2: Shrink (newSize < size)
                if (newSize < size) {
                    // We only delete the objects that are *removed* from the end.
                    // This assumes T is a pointer and we own the memory.
                    for (size_t i = newSize; i < size; ++i) {
                        delete data[i];     // Free the pointed-to memory
                        data[i] = nullptr;  // Set the pointer to NULL for safety
                    }
                }

                // Case 3: Grow within capacity (newSize > size, newSize <= capacity)
                if (newSize > size) {
                    // The new slots (from size to newSize - 1) need to be initialized.
                    // Since T is assumed to be Point*, these new slots should be set to nullptr.
                    for (size_t i = size; i < newSize; ++i) {
                        data[i] = nullptr;
                    }
                }

                size = newSize;
            }

            inline T& operator[](size_t index) {
                if (index > maxCall) maxCall = index;
                if (index >= capacity)
                    resize(index * 2);
                return data[index];
            }
            inline T& at(size_t index) {
                if (index >= capacity)
                    resize(index * 2);

                return data[index];
            }
            void clear() {
                size = 0;
            }
        };

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
			~myvector();
			void reserve(int newsize);

			int size();

			void resize(int newSize);
			inline T& operator[](const int index) {
				return vec[index >> shift][index & mask];
			}

			inline const T& at(const int index) {
				return vec[index >> shift][index & mask];
			}

			void clear();
		};
    }
}
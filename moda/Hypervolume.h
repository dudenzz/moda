#pragma once
#include "Helpers.h"
#include "Point.h"
namespace moda {
    namespace backend {
        class Backend {
        public: 
            static DType Hypervolume(const Point* nadirPoint, const Point* p2, const Point* idealPoint, int numberOfObjectives);
            static DType Hypervolume(const Point* nadirPoint, const Point* idealPoint, int numberOfObjectives);
            static short off(short offset, short j, int numberOfObjectives);


            static void Normalize(std::vector <DType>& p, int numberOfObjectives);
            static DType Norm(std::vector <DType>& p, int numberOfObjectives);

        };
    }
}
#pragma once
#include "Helpers.h"
#include "Point.h"
namespace moda {
    namespace backend {
        DType Hypervolume(const Point* nadirPoint, const Point* p2, const Point* idealPoint, int numberOfObjectives);
        DType Hypervolume(const Point* nadirPoint, const Point* idealPoint, int numberOfObjectives);
        short off(short offset, short j, int numberOfObjectives);


        void Normalize(std::vector <DType>& p, int numberOfObjectives);
        DType Norm(std::vector <DType>& p, int numberOfObjectives);


    }
}
#include "Hypervolume.h"
/*************************************************************************

 Improved Quick Hypervolume Computation (IQHV_for_hss)

 ---------------------------------------------------------------------

                           Copyright (C) 2025
          Andrzej Jaszkiewicz <ajaszkiewicz@cs.put.poznan.pl>
          Piotr Zielniewicz <piotr.zielniewicz@cs.put.poznan.pl>
          Jakub Dutkiewicz <jakub.dutkiewicz@put.poznan.pl>

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at https://mozilla.org/MPL/2.0/.

 ----------------------------------------------------------------------

 Relevant literature:

 [1]  Andrzej Jaszkiewicz and Piotr Zielniewicz. 2023.
      On-line Quick Hypervolume Algorithm.
      In Proceedings of the
      Companion Conference on Genetic and Evolutionary Computation (GECCO '23 Companion).
      Association for Computing Machinery,
      New York, NY, USA, 371–374. https://doi.org/10.1145/3583133.3590650



*************************************************************************/

namespace moda {
    namespace backend {

            short Backend::off(short offset, short j, int numberOfObjectives)
            {
                return (j + offset) % numberOfObjectives;
            }

            DType Backend::Hypervolume(const Point* nadirPoint, const Point* p2, const Point* idealPoint, int numberOfObjectives)
            {

                DType s = 1;
                short j;
                for (j = 0; j < numberOfObjectives; j++)
                {
                    if (p2->ObjectiveValues[j] > idealPoint->ObjectiveValues[j])
                        s *= idealPoint->ObjectiveValues[j] - nadirPoint->ObjectiveValues[j];
                    else
                        s *= p2->ObjectiveValues[j] - nadirPoint->ObjectiveValues[j];

                }
                return s;
            }
            DType Backend::Hypervolume(const Point* nadirPoint, const Point* idealPoint, int numberOfObjectives) {
                DType s = 1;
                short j;
                for (j = 0; j < numberOfObjectives; j++) {

                    s *= (idealPoint->ObjectiveValues[j] - nadirPoint->ObjectiveValues[j]);
                }
                return s;
            }


            void Backend::Normalize(std::vector <DType>& p, int numberOfObjectives) {
                DType nrm = Norm(p, numberOfObjectives);
                DType s = 0;
                int j;
                for (j = 0; j < numberOfObjectives; j++) {
                    p[j] /= nrm;
                }
            }

            DType Backend::Norm(std::vector <DType>& p, int numberOfObjectives) {
                DType s = 0;
                int j;
                for (j = 0; j < numberOfObjectives; j++) {
                    s += p[j] * p[j];
                }
                return sqrt(s);
            }
        }

    

}
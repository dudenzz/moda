#include "Point.h"
namespace moda {
    // Constructors
    Point::Point()
    {
        //ObjectiveValues = new DType[MAXOBJECTIVES];
        NumberOfObjectives = MAXOBJECTIVES;
    }

    Point::Point(int NumberOfObjectives)
    {
        this->NumberOfObjectives = NumberOfObjectives;
        //ObjectiveValues = new DType[NumberOfObjectives];
    }
    //
    Point Point::ones(int NumberOfObjectives)
    {
        Point onesPoint = Point(NumberOfObjectives);
        for (int i = 0; i < NumberOfObjectives; i++)
            onesPoint[i] = 1;
        return onesPoint;
    }

    Point Point::zeroes(int NumberOfObjectives)
    {
        Point zeroesPoint = Point(NumberOfObjectives);
        for (int i = 0; i < NumberOfObjectives; i++)
            zeroesPoint[i] = 0;
        return zeroesPoint;
    }


     //Copy Constructor

    Point::Point(const Point& Point)
    {
        
        NumberOfObjectives = Point.NumberOfObjectives;
        if (ObjectiveValues == NULL)
        {
            //ObjectiveValues = new DType[NumberOfObjectives] ; //CHECKED
        }
        for (int i = 0; i < NumberOfObjectives; i++)
            ObjectiveValues[i] = Point.ObjectiveValues[i];
    }


    /** Copy operator */
    Point& Point::operator=(Point &Point)
    {
        NumberOfObjectives = Point.NumberOfObjectives;
        //ObjectiveValues = new DType[NumberOfObjectives];
        for (int i = 0; i < NumberOfObjectives; i++)
            this->ObjectiveValues[i] = Point.ObjectiveValues[i];
        return *this;
    }


    Point& Point::operator=(const Point& Point)
    {
        NumberOfObjectives = Point.NumberOfObjectives;
        //ObjectiveValues = new DType[NumberOfObjectives];
        for (int i = 0; i < NumberOfObjectives; i++)
            this->ObjectiveValues[i] = Point.ObjectiveValues[i];
        return *this;
    }

    /** getter */
    DType Point::operator[](int n) const
    {
        return ObjectiveValues[n];
    }
    /** getter */
    DType Point::get(int n) const
    {
        return ObjectiveValues[n];
    }
    /** setter */
    DType& Point::operator[](int n)
    {
        return ObjectiveValues[n];
    }
    /*DType& Point<DType>::operator[](int n)
    {
        return ObjectiveValues[n];
    }*/
    /** getter */
    Point& Point::operator-(DType value)
    {
        Point newPoint = Point(*this);

        for (int i = 0; i < NumberOfObjectives; i++)
        {
            (newPoint)[i] = ObjectiveValues[i] - value;
        }
        return newPoint;
    }
    Point& Point::operator+(DType value)
    {
        Point newPoint = Point(*this);

        for (int i = 0; i < NumberOfObjectives; i++)
        {
            (newPoint)[i] = ObjectiveValues[i] + value;
        }
        return newPoint;
    }


    /** Reads the point from the stream */
    std::istream& Point::Load(std::istream& Stream)
    {
        int i;
        for (i = 0; i < NumberOfObjectives; i++)
        {
            Stream >> ObjectiveValues[i];
            //ObjectiveValues[i] = ObjectiveValues[i];
        }
        char c;
        do
        {
            Stream.get(c);
        } while (c != '\n' && c > 0);
        return Stream;
    }


    std::istream& Point::operator<<(std::istream& Stream)
    {
        return this->Load(Stream);
    }


    /** Saves objective values to an open Stream
     *
     *	Values are separated by TAB character */
    std::ostream& Point::Save(std::ostream& Stream)
    {
        int i;
        for (i = 0; i < NumberOfObjectives; i++)
        {
            Stream << ObjectiveValues[i];
            Stream << '\x09';
        }
        Stream << '\x09';
        return Stream;
    }
    /**
     * Compares this point (LHS) with a given point (RHS). Returns one of the TCompare results:
     * - Dominating - if ALL of the objective values in the LHS point are greater than objective values of the RHS point
     * - Dominated - if ALL of the objective values in the RHS point are greater than objective values of the LHS point
     * - Non-dominated - points are not equal, and niether one weakly_dominates the other
     * - Equal-sol - points are equal
     *
     * @param point point to compare with (RHS).
     */

    ComparisonResult Point::Compare(Point& point, bool maximization)
    {
        bool bBetter = false;
        bool bWorse = false;

        short i = 0;
        do
        {
            if (maximization)
            {
                if (ObjectiveValues[i] > point.ObjectiveValues[i])
                    bBetter = true;
                if (point.ObjectiveValues[i] > ObjectiveValues[i])
                    bWorse = true;
                i++;
            }
            else
            {
                if (ObjectiveValues[i] < point.ObjectiveValues[i])
                    bBetter = true;
                if (point.ObjectiveValues[i] < ObjectiveValues[i])
                    bWorse = true;
            }
        } while (!(bWorse && bBetter) && (i < NumberOfObjectives));

        if (bWorse)
        {
            if (bBetter)
            {
                return _Nondominated;
            }
            else
            {
                return _Dominated;
            }
        }
        else
        {
            if (bBetter)
            {
                return _Dominating;
            }
            else
            {
                return _EqualSol;
            }
        }
    }


    std::ostream& operator<<(std::ostream& os, Point dt)
    {
        return dt.Save(os);
    };

 /*   template<typename T> Point<T>& operator-(float value, Point<float> p) {
        Point<T>* newPoint = new Point(p);
        newPoint->NumberOfObjectives = p.NumberOfObjectives;
        for (int i = 0; i < p.NumberOfObjectives; i++)
        {
            (*newPoint)[i] = value - p[i];
        }
        return *newPoint;
    }*/
    /*Point<DType>& operator-(DType value, Point<DType> p) {
        Point<DType>* newPoint = new Point(p);
        newPoint->NumberOfObjectives = p.NumberOfObjectives;
        for (int i = 0; i < p.NumberOfObjectives; i++)
        {
            (*newPoint)[i] = value - p[i];
        }
        return *newPoint;
    }*/

    DType Point::Distance(Point& ComparedPoint, Point& IdealPoint, Point& NadirPoint) {
        DType s = 0;
        int iobj; for (iobj = 0; iobj < NumberOfObjectives; iobj++) {
            DType Range = IdealPoint.ObjectiveValues[iobj] - NadirPoint.ObjectiveValues[iobj];
            if (Range == 0)
                Range = 1;
            Range = 1;
            DType s1 = (ObjectiveValues[iobj] - ComparedPoint.ObjectiveValues[iobj]) / Range;
            s += s1 * s1;
        }
        return sqrt(s);
    }

    extern unsigned long long scalarizingCalls(0);

    DType Point::CleanChebycheffScalarizingFunctionInverse(std::vector<DType>& weightVector, Point& referencePoint) {
        scalarizingCalls++;
        DType min = 1e30;
        short i;
        for (i = 0; i < NumberOfObjectives; i++) {
            DType s = weightVector[i] * (referencePoint.ObjectiveValues[i] - ObjectiveValues[i]);
            if (s < min)
                min = s;
        }

        return min;
    }
    DType Point::CleanChebycheffScalarizingFunctionOriginal(std::vector<DType>& weightVector, Point& referencePoint) {
        
        DType max = -1e30;
        short i;
        for (i = 0; i < NumberOfObjectives; i++) {
            DType s = weightVector[i] * (referencePoint.ObjectiveValues[i] - ObjectiveValues[i]);
            if (s > max)
                max = s;
        }

        return max;
    }
    Point& operator-(DType value, Point  p) {
        Point* newPoint = new Point(p);
        newPoint->NumberOfObjectives = p.NumberOfObjectives;
        for (int i = 0; i < p.NumberOfObjectives; i++)
        {
            (*newPoint)[i] = value - p[i];
        }
        return *newPoint;
    }
}
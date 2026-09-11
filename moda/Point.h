#ifndef C_POINT
#define C_POINT
#include "include.h"
#include "Helpers.h"
//zmiana namzwy namespacu z qhv na iqhv
namespace moda
{
	extern unsigned long long scalarizingCalls;
	/** Point in objective space */
	class Point
	{
	public:
		
		int NumberOfObjectives;
		DType ObjectiveValues[MAXOBJECTIVES];

		/** Constructors */
		Point();
		Point(std::vector<DType> ObjectiveValues);
		Point(int NumberOfObjectives);
		static Point ones(int NumberOfObjectives);
		static Point elevens(int NumberOfObjectives);
		static Point negElevens(int NumberOfObjectives);
		static Point zeroes(int NumberOfObjectives);
		/** Copy constructor */
		Point(const Point& Point);
		/** Copy operator */
		Point& operator=(Point& Point);
		Point& operator=(const Point& Point);
		Point& operator-(DType value);
		Point& operator-();
		Point& operator+(DType value);
		bool operator>(const Point point) const;
		bool operator<(const Point point) const;
		bool operator>>(const Point point) const;
		bool operator<<(const Point point) const;
		bool operator>(std::vector<Point*> point) const;
		bool operator>>(std::vector<Point*> point);
		inline DType contribution(moda::Point& idealPoint, moda::Point& nadirPoint) {
			DType contribution = 1.0;
			for (int i = 0; i < this->NumberOfObjectives; i++) {
				DType max = std::max(this->ObjectiveValues[i], idealPoint.ObjectiveValues[i]);
				DType min = std::min(this->ObjectiveValues[i], nadirPoint.ObjectiveValues[i]);
				contribution *= max - min;
			}
			return contribution;
		}
		/** Comparator */
		
		ComparisonResult Compare(Point& point, bool maximization);
		/** Getter operator */
		DType operator[](int n) const;
		/** Getter */
		DType get(int n) const;
		/** Setter operator */
		DType& operator[](int n);
		/** Reads the point from the stream */
		std::istream& Load(std::istream& Stream);
		std::istream& operator<<(std::istream& os);
		/** Saves objective values to an open Stream
		 *	Values are separated by TAB character */
		std::ostream& Save(std::ostream& Stream);

		DType Distance(Point& ComparedPoint, Point& IdealPoint, Point& NadirPoint);

		DType CleanChebycheffScalarizingFunctionInverse(std::vector<DType>& weightVector, Point& referencePoint);

		DType CleanChebycheffScalarizingFunctionOriginal(std::vector<DType>& weightVector, Point& referencePoint);


	private:
		/** Vector of objective values */
	};

	std::ostream& operator<<(std::ostream& os, Point dt);
	Point& operator-(DType value, Point  p);
}
#endif
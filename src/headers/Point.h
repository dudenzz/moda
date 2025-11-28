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
		Point(int NumberOfObjectives);
		static Point ones(int NumberOfObjectives);
		static Point zeroes(int NumberOfObjectives);
		/** Destructor **/
		virtual ~Point();
		/** Copy constructor */
		Point(const Point& Point);
		/** Copy operator */
		Point& operator=(Point& Point);
		Point& operator=(const Point& Point);
		Point& operator-(DType value);
		Point& operator+(DType value);
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
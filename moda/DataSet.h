#pragma once
#include "include.h"
#include "DataSetParameters.h"
#include "Point.h"
#include "SolutionContainer.h"
#include "NDTree.h"
#include <sstream>

namespace moda {
	/// <summary>
	/// This class represents an optimization problem, a dataset. Dataset is comprised of a set of points. 
	/// </summary>
	
	class DataSet : SolutionContainer {
	public:

#pragma region public data
		/// <summary>
		/// A set of points in a dataset. This is public due to the nature of memory management. In future, it should be moved to private.
		/// </summary>
		std::vector<Point*> points;
		/// <summary>
		/// full filename. In future this should be moved to private.
		/// </summary>
		std::string filename;
		Point* operator[](int n) const;
		Point& operator[](int n);
		Point* get(int n) const;
#pragma endregion
#pragma region Optimization Types
		/// Optimization types - minimization and maximization; maximization is a default optimization type
		enum OptimizationType {minimization, maximization};
		/// <summary>
		/// Type of this problem. Maximalization by default.
		/// </summary>
		OptimizationType typeOfOptimization;
#pragma endregion
#pragma region Constructors	
		/// <summary>
		/// Non-parametrized constructor
		/// </summary>
		DataSet(int nObjectives = 2);
		/// <summary>
		/// Copy constructor
		/// </summary>
		DataSet(const DataSet& dataset);
		/// <summary>
		/// This constructor will read the data from a file. If the name is standardized (i.e. name_of_experiment_dXXX_nXXX_ZZ) the dataset properties will be gathered from the filename.
		/// </summary>
		/// <param name="filename">Full path to the file with data</param>
		/// <param name="normalizedName">Is the filename standardized(i.e. name_of_experiment_dXXX_nXXX_ZZ)</param>
		DataSet(const std::string filename, bool normalizedName);
		/// <summary>
		/// This constructor will read the data from a file. Dataset properties are provided to this constructor
		/// </summary>
		/// <param name="filename">Full path to the file</param>
		/// <param name="parameters">Dataset properties</param>
		DataSet(const std::string filename, DataSetParameters settings);
		/// <summary>
		/// This constructor will read the data from a file. Dataset properties are provided to this constructor
		/// </summary>
		/// <param name="filename">Full path to the filename</param>
		/// <param name="name">Name of the experiment (misc)</param>
		/// <param name="dimensions">number of objectives</param>
		/// <param name="sample">number of a sample in the experiment</param>
		/// <param name="nPoints">number of points in the file</param>
		DataSet(const std::string filename, std::string name, int dimensions, int sample, int nPoints);
		/** Destructor **/
		virtual ~DataSet();
		/** Copy constructor */
		// TOptimizationProblem(TOptimizationProblem& problem);
#pragma endregion
#pragma region Accessors
		//Getters
		/// get calculated ideal point for this dataset
		Point* getIdeal();
		/// set ideal point for this dataset
		void setIdeal(Point*);
		// get caluclated nadir for this dataset
		Point* getNadir();
		/// set nadir point for this dataset
		void setNadir(Point*);
		// get properties for this dataset
		DataSetParameters* getParameters();
		//setters
		/// <summary>
		/// set properties for this dataset
		/// </summary>
		/// <param name="parameters">dataset properties object</param>
		void setParameters(DataSetParameters settings);
		/// <summary>
		/// set number of objectives
		/// </summary>
		/// <param name="dim">number of objectives</param>
		void setDimensionality(int dim);
		/// <summary>
		/// set experiment name for this dataset
		/// </summary>
		/// <param name="name">experiment name</param>
		void setName(std::string name);
		/// <summary>
		/// set number of points in this dataset.
		/// </summary>
		/// <param name="npts"></param>
		void setNumberOfPoints(int npts);
		/// <summary>
		/// set sample number in this experiment
		/// </summary>
		/// <param name="sampleN">sample number</param>
		void setSampleNumber(int sampleN);
		/// <summary>
		/// Should this dataset be normalized?
		/// </summary>

#pragma endregion
#pragma region Stream operations
		/// <summary>
		/// read dataset from stream
		/// </summary>
		/// <param name="stream">stream</param>
		/// <returns>stream</returns>
		std::istream& Load(std::istream& stream);

		void Save(const std::string filename);
		static DataSet *LoadFromFilename(const std::string filename);
		static std::vector<DataSet*> LoadBulk(const std::string directory);
		std::string to_string();

		NDTree<Point> toNDTree();

#pragma endregion
#pragma region operators
		/** Copy operator */
		DataSet& operator = (DataSet& problem);
#pragma endregion
#pragma region Functions
		/// <summary>
		/// normalization function callback
		/// </summary>
		std::vector<Point*>(*NormalizingFunction)(std::vector<Point*> points);
		/// <summary>
		/// default normalization function (column wise min-max normalization with the 0.0 - 1.0 range)
		/// </summary>
		/// <param name="points">set of points</param>
		/// <returns>normalized set of point</returns>
		std::vector<Point*> defaultScalingFuncation(std::vector<Point*> points);
		void reverseObjectives();
		/// <summary>
		/// Normalize the dataset
		/// </summary>
		void normalize();
		/// <summary>
		/// A function, which calculates nadir - 0.1 * (ideal - nadir). A default worse reference point.
		/// </summary>
		/// <returns></returns>

		bool add(Point* point);
		bool remove(Point* point);
		Point* remove(int i);
		void clear();
		void RemoveDominated();
		Point* defaultWorsePoint();
		Point* defaultBetterPoint();

		
#pragma endregion
#pragma region grouping

		/// <summary>
		/// Group a vector of Datasets according to a given criteria.
		/// </summary>
		/// <param name="problems">datasets</param>
		/// <param name="grouping">grouping criteria</param>
		/// <returns>grouped datasets</returns>
		static std::vector<std::vector<DataSet>> BulkGroup(std::vector<DataSet> problems, ProblemGrouping grouping);
		/// <summary>
		/// Print details of a grouped vector of datasets
		/// </summary>
		/// <param name="groupedProblems">vector of datasets</param>
		static void printGroupsDetails(std::vector<std::vector<DataSet>> groupedProblems);
		
#pragma endregion
	private:
#pragma region private data
		Point* nadir;
		Point* ideal;
		bool presetParameters;
		DataSetParameters parameters;
#pragma endregion
#pragma region private functions
		void calculateNadir();
		void calculateIdeal();
#pragma endregion
	};
}

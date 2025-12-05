#pragma once
#include "include.h"
#include "DataSetParameters.h"




#include "Result.h"







namespace moda {

	/** Possible relations between points in multiple objective space */
	enum ComparisonResult { _Dominating, _Dominated, _Nondominated, _EqualSol };
	/// <summary>
	/// Types of dataset grouping
	/// </summary>
	enum ProblemGrouping { Name, Dimensionality, NameDimensionality };

	/*simple string functions - split and join*/
	std::vector<std::string> split(std::string& s, const std::string& delimiter);
	std::string join(const std::vector<std::string>& lst, const std::string& delimiter);
	
	/**dedicated filename parser, which gathers problem parameters from normalized filename
	 * format - path/to/file/experiment_name_dXXXnYYY_ZZ
	 * XXX - number of dimensions
	 * YYY - number of points in the problem
	 * ZZ - number of the problem for this experiment
	*/
	std::tuple<int, int> parseSettings(const std::string settingsString);

	/* Count files in a given directory*/
	int countFilesInDirectory(std::string directory);

	/* Fancy std output */

	/* progress bar */
	void make_progress(float progress, int n1, int n2, std::string message = "");

	/* default solver callbacks */
	void ProgressBarCallback(int currentIteration, int totalIterations, Result *stepResult);

	void DefaultStartCallback(DataSetParameters problemSettings, std::string solverMessage);
	void DefaultEndCallback(DataSetParameters problemSettings, Result *finalResult);
	void FileSaveEndCallback(DataSetParameters problemSettings, Result* finalResult);


	// trim from end of string (right)
	inline std::string& rtrim(std::string& s, const char* t = " \t\n\r\f\v")
	{
		s.erase(s.find_last_not_of(t) + 1);
		return s;
	}

	// trim from beginning of string (left)
	inline std::string& ltrim(std::string& s, const char* t = " \t\n\r\f\v")
	{
		s.erase(0, s.find_first_not_of(t));
		return s;
	}

	// trim from both ends of string (right then left)
	inline std::string& trim(std::string& s, const char* t = " \t\n\r\f\v")
	{
		return ltrim(rtrim(s, t), t);
	}

	DType logarithm(DType param, DType base);
}

#include "SolverParameters.h"
namespace moda
{
	QR2Parameters::QR2Parameters(ReferencePointCalculationStyle worseReferencePointCalculationStyle, ReferencePointCalculationStyle betterReferencePointCalculationStyle,
		int MaxEstimationTime, bool callbacks, bool calculateHV) :SolverParameters(worseReferencePointCalculationStyle, betterReferencePointCalculationStyle, MaxEstimationTime, callbacks)
	{
		CalculateHV = calculateHV;
	}
	QHV_BRParameters::QHV_BRParameters(ReferencePointCalculationStyle worseReferencePointCalculationStyle, ReferencePointCalculationStyle betterReferencePointCalculationStyle,
		int MaxEstimationTime, bool callbacks) :SolverParameters(worseReferencePointCalculationStyle, betterReferencePointCalculationStyle, MaxEstimationTime, callbacks)
	{

	}
	MCHVParameters::MCHVParameters(ReferencePointCalculationStyle worseReferencePointCalculationStyle, ReferencePointCalculationStyle betterReferencePointCalculationStyle,
		int MaxEstimationTime, bool callbacks) :SolverParameters(worseReferencePointCalculationStyle, betterReferencePointCalculationStyle, MaxEstimationTime, callbacks)
	{

	}
	IQHVParameters::IQHVParameters(ReferencePointCalculationStyle worseReferencePointCalculationStyle, ReferencePointCalculationStyle betterReferencePointCalculationStyle,
		int MaxEstimationTime, bool callbacks) :SolverParameters(worseReferencePointCalculationStyle, betterReferencePointCalculationStyle, MaxEstimationTime, callbacks)
	{

	}
	QHV_BQParameters::QHV_BQParameters(ReferencePointCalculationStyle worseReferencePointCalculationStyle, ReferencePointCalculationStyle betterReferencePointCalculationStyle,
		int MaxEstimationTime,  bool callbacks, SwitchParameters mcSettings, bool TurnOnMonteCarlo) :SolverParameters(worseReferencePointCalculationStyle, betterReferencePointCalculationStyle, MaxEstimationTime, callbacks)
	{
		this->SwitchToMCSettings = mcSettings;
		MonteCarlo = TurnOnMonteCarlo;
	}
 DBHVEParameters::DBHVEParameters(ReferencePointCalculationStyle worseReferencePointCalculationStyle, ReferencePointCalculationStyle betterReferencePointCalculationStyle,
		int MaxEstimationTime, bool callbacks) :SolverParameters(worseReferencePointCalculationStyle, betterReferencePointCalculationStyle, MaxEstimationTime,  callbacks)
	{
		
	}
	HSSParameters::HSSParameters(ReferencePointCalculationStyle worseReferencePointCalculationStyle, ReferencePointCalculationStyle betterReferencePointCalculationStyle,
		int MaxEstimationTime, bool callbacks, HSSParameters::SubsetSelectionStrategy SubsetSelectionQHVIncremental, HSSParameters::StoppingCriteriaType StoppingCriteria, int StopTime, int StopSize) : SolverParameters(worseReferencePointCalculationStyle, betterReferencePointCalculationStyle, MaxEstimationTime, callbacks)
	{
		this->Strategy = SubsetSelectionQHVIncremental;
		this->StoppingCriteria = StoppingCriteria;
		this->StoppingSubsetSize = StopTime;
		this->StoppingTime = StopSize;
	}
	QEHCParameters::QEHCParameters(ReferencePointCalculationStyle worseReferencePointCalculationStyle, ReferencePointCalculationStyle betterReferencePointCalculationStyle,
		int MaxEstimationTime, bool callbacks, unsigned long int iterationsLimit, bool sort,
		QEHCParameters::SearchSubjectOption minimalContribution, bool shuffle, int offset) : SolverParameters(worseReferencePointCalculationStyle, betterReferencePointCalculationStyle,MaxEstimationTime,callbacks) {
		this->iterationsLimit = iterationsLimit;
		this->sort = sort;
		this->SearchSubject = minimalContribution;
		this->shuffle = shuffle;
		this->offset = offset;
	}

	Point* SolverParameters::GetBetterReferencePoint(DataSet *dataset)
	{
		Point* ide;
		switch (BetterReferencePointCalculationStyle)
		{
		case ReferencePointCalculationStyle::tenpercent:
			return dataset->defaultBetterPoint();
		case ReferencePointCalculationStyle::userdefined:
			return betterReferencePoint;

		case ReferencePointCalculationStyle::zeroone:
			return new Point(Point::ones(dataset->getParameters()->NumberOfObjectives));
		case ReferencePointCalculationStyle::exact:
			ide = dataset->getIdeal();
			return new Point(*ide);
		case ReferencePointCalculationStyle::epsilon:
		default:
			Point t =  *(dataset->getIdeal()) + EPSILON;
			return new Point(t);
		}
	}

	Point* SolverParameters::GetWorseReferencePoint(DataSet *dataset)
	{
		switch (WorseReferencePointCalculationStyle)
		{

		case ReferencePointCalculationStyle::tenpercent:
			return dataset->defaultWorsePoint();
		case ReferencePointCalculationStyle::userdefined:
			return worseReferencePoint;

		case ReferencePointCalculationStyle::zeroone:
			return new Point(Point::zeroes(dataset->getParameters()->NumberOfObjectives));
		case ReferencePointCalculationStyle::exact:
			return new Point(*dataset->getNadir());
		case ReferencePointCalculationStyle::epsilon:
		default:
			Point t = *(dataset->getNadir()) - EPSILON;
			return new Point(t);
		}
	}
	SolverParameters::SolverParameters(ReferencePointCalculationStyle worseReferencePointCalculationStyle, ReferencePointCalculationStyle betterReferencePointCalculationStyle,
		int MaxEstimationTime, bool callbacks)
	{

		this->callbacks = callbacks;
		this->MaxEstimationTime = MaxEstimationTime;
		BetterReferencePointCalculationStyle = betterReferencePointCalculationStyle;
		WorseReferencePointCalculationStyle = worseReferencePointCalculationStyle;
		betterReferencePoint = NULL;
		worseReferencePoint = NULL;
		
		


	}

	SolverParameters::SolverParameters() {
		MaxEstimationTime = 1000;
		callbacks = false;
		BetterReferencePointCalculationStyle = ReferencePointCalculationStyle::userdefined;
		WorseReferencePointCalculationStyle = ReferencePointCalculationStyle::userdefined;
		betterReferencePoint = NULL;
		worseReferencePoint = NULL;
	}




}
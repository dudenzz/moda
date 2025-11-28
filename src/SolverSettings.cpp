#include "SolverSettings.h"
namespace moda
{
	QR2Parameters::QR2Parameters(DataSet set, LowerReferencePointCalculationStyle referencePointsCalculationStyle,
		int MaxEstimationTime, int SaveInterval, bool callbacks) :SolverParameters(set, referencePointsCalculationStyle, MaxEstimationTime, SaveInterval, callbacks)
	{

	}
	QHV_BRParameters::QHV_BRParameters(DataSet set, LowerReferencePointCalculationStyle referencePointsCalculationStyle,
		int MaxEstimationTime, int SaveInterval, bool callbacks) :SolverParameters(set, referencePointsCalculationStyle, MaxEstimationTime, SaveInterval, callbacks)
	{

	}
	MCHVParameters::MCHVParameters(DataSet set, LowerReferencePointCalculationStyle referencePointsCalculationStyle,
		int MaxEstimationTime, int SaveInterval, bool callbacks) :SolverParameters(set, referencePointsCalculationStyle, MaxEstimationTime, SaveInterval, callbacks)
	{

	}
	IQHVParameters::IQHVParameters(DataSet set, LowerReferencePointCalculationStyle referencePointsCalculationStyle,
		int MaxEstimationTime, int SaveInterval, bool callbacks) :SolverParameters(set, referencePointsCalculationStyle, MaxEstimationTime, SaveInterval, callbacks)
	{

	}
	QHV_BQParameters::QHV_BQParameters(DataSet set, LowerReferencePointCalculationStyle referencePointsCalculationStyle,
		int MaxEstimationTime, int SaveInterval, bool callbacks, SwitchSettings mcSettings, bool TurnOnMonteCarlo) :SolverParameters(set, referencePointsCalculationStyle, MaxEstimationTime, SaveInterval, callbacks)
	{
		this->SwitchToMCSettings = mcSettings;
		MonteCarlo = TurnOnMonteCarlo;
	}
	HVEParameters::HVEParameters(DataSet set, LowerReferencePointCalculationStyle referencePointsCalculationStyle,
		int MaxEstimationTime, int SaveInterval, bool callbacks) :SolverParameters(set, referencePointsCalculationStyle, MaxEstimationTime, SaveInterval, callbacks)
	{
		this->SaveInterval = SaveInterval;
	}
	HSSParameters::HSSParameters(DataSet set, LowerReferencePointCalculationStyle referencePointsCalculationStyle,
		int MaxEstimationTime, int SaveInterval, bool callbacks, bool SubsetSelectionQHVIncremental, int StoppingCriteria, int StopTime, int StopSize) : SolverParameters(set, referencePointsCalculationStyle, MaxEstimationTime, SaveInterval, callbacks)
	{
		this->SubsetSelectionQHVIncremental = SubsetSelectionQHVIncremental;
		this->StoppingCriteria = StoppingCriteria;
		this->StoppingSubsetSize = StopTime;
		this->StoppingTime = StopSize;
	}
	QEHCParameters::QEHCParameters(DataSet set, LowerReferencePointCalculationStyle referencePointsCalculationStyle,
		int MaxEstimationTime, int SaveInterval, bool callbacks, unsigned long int iterationsLimit, bool sort,
		bool minimalContribution, bool shuffle, int offset) : SolverParameters(set,referencePointsCalculationStyle,MaxEstimationTime,SaveInterval,callbacks) {
		this->iterationsLimit = iterationsLimit;
		this->sort = sort;
		this->minimalContribution = minimalContribution;
		this->shuffle = shuffle;
		this->offset = offset;
	}
	SolverParameters::SolverParameters(DataSet dataset, LowerReferencePointCalculationStyle referencePointsCalculationStyle,
		int MaxEstimationTime, int SaveInterval, bool callbacks)
	{

		this->callbacks = callbacks;
		
		this->SaveInterval = SaveInterval;
		this->MaxEstimationTime = MaxEstimationTime;

		
		
		Point zeroes = Point::zeroes(dataset.getParameters()->NumberOfObjectives);;
		Point ones = Point::ones(dataset.getParameters()->NumberOfObjectives);;
		this->referencePointsCalculationStyle = referencePointsCalculationStyle;
		if(dataset.getParameters()->normalize) dataset.normalize();
		switch (referencePointsCalculationStyle)
		{

		case LowerReferencePointCalculationStyle::tenpercent:
			betterReferencePoint = dataset.getIdeal();
			worseReferencePoint = dataset.defaultWorsePoint();
			break;
		case LowerReferencePointCalculationStyle::userdefined:
			break;
		case LowerReferencePointCalculationStyle::zeroone:
			if (dataset.typeOfOptimization == DataSet::OptimizationType::minimalization)
			{
				worseReferencePoint = new Point(ones);
				betterReferencePoint = new Point(zeroes);
			}
			else
			{
				worseReferencePoint = new Point(zeroes);
				betterReferencePoint = new Point(ones);
			}
			break;
		case LowerReferencePointCalculationStyle::exact:
			worseReferencePoint = dataset.getNadir();
			betterReferencePoint = dataset.getIdeal();
			break;
		case LowerReferencePointCalculationStyle::epsilon:
		default:
			betterReferencePoint = dataset.getIdeal();
			Point t = *(dataset.getNadir()) - EPSILON;
			worseReferencePoint = new Point(t);
		}

	}




}
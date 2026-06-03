Problem solvers
=====

.. _tutorials_solvers:

.. figure:: images/solvers_umls.png
   :width: 600
   :align: center

   Solvers Class Diagram.

.. _moda-solver:

Solver Class
============

.. cpp:namespace:: moda

.. cpp:class:: Solver

   The base class providing an interface and utility methods for solving optimization problems.

   .. cpp:member:: DataSet* currentlySolvedProblem

      Pointer to the dataset currently being processed.

   .. cpp:member:: DataSetParameters* currentSettings

      Pointer to the configuration settings for the current problem.

   Callbacks
   ---------

   The class supports hooks for monitoring the execution process.

   .. cpp:member:: void (*StartCallback)(DataSetParameters problemSettings, std::string SolverMessage)

      Called when the solver starts execution.

   .. cpp:member:: void (*IterationCallback)(int currentIteration, int totalIterations, Result* stepResult)

      Called at the end of each iteration to report progress.

   .. cpp:member:: void (*EndCallback)(DataSetParameters problemSettings, Result* stepResult)

      Called upon the completion of the solving process.

   Public Methods
   --------------

   .. cpp:function:: virtual Result* Solve(DataSet* problem, SolverParameters settings)

      Abstract function. Each solver implements it to solve the problem and return a result.

   Protected Methods
   -----------------

   .. cpp:function:: void prepareData(DataSet* problem, SolverParameters settings)

      Prepares operational data structures before starting the solver.

.. cpp:namespace:: moda

Solver Settings Reference
=========================

The `moda` library utilizes a hierarchical configuration system for various optimization solvers. All specific solver settings inherit from the base :cpp:class:`SolverParameters` class.

Base Solver Parameters Class
-----------------

.. cpp:class:: SolverParameters

   The foundational class for all solver configurations.

   .. cpp:enum:: ReferencePointCalculationStyle

    Defines how reference points are calculated.

    .. cpp:enumerator:: epsilon 
        
        For the dataset :math:`X \subseteq \mathbb{R}^n`, define the minimum and maximum values for each dimension :math:`i` as:

        .. math::

        m_i = \min_{x \in X} (x_i), \quad M_i = \max_{x \in X} (x_i)

        The adjusted reference points are defined as:

        .. math::

        { m_i - \epsilon \mid 0 < i \le n \}

        { M_i + \epsilon \mid 0 < i \le n \}

        .. note::
        The parameter :math:`\epsilon` is currently set to a default value of **0.001**. 
        You can modify this value by updating the corresponding constant in the ``include.h`` file.



    .. cpp:enumerator:: tenpercent

        For the dataset :math:`X \subseteq \mathbb{R}^n`, define the minimum and maximum values for each dimension :math:`i` as:

        .. math::

        m_i = \min_{x \in X} (x_i), \quad M_i = \max_{x \in X} (x_i)

        The adjusted reference points are defined as:

        .. math::

        { m_i - 0.1 \cdot |m_i - M_i| \mid 0 < i \le n \}

        { M_i + 0.1 \cdot |m_i - M_i| \mid 0 < i \le n \}

    .. cpp:enumerator:: zeroone

        For the dataset :math:`X \subseteq \mathbb{R}^n`, the reference points are defined as:

        .. math::

        \{ 0 \mid 0 < i \le n \}

        \{ 1 \mid 0 < i \le n \} .

    .. cpp:enumerator:: userdefined

        The reference points are defined by user prior to the solver execution. In order to use this setting declare the worseReferencePoint and betterReferencePoint values.

    .. cpp:enumerator:: exact

        For the dataset :math:`X \subseteq \mathbb{R}^n`, the reference points are defined as:

        .. math::

        \{ \min_{x \in X} (x_i) \mid 0 < i \le n \}

        \{ \max_{x \in X} (x_i) \mid 0 < i \le n \} .

      .. cpp:enumerator:: pymoo

        For the dataset :math:`X \subseteq \mathbb{R}^n`, define the minimum and maximum values for each dimension :math:`i` as:

        .. math::

        m_i = \min_{x \in X} (x_i), \quad M_i = \max_{x \in X} (x_i)

        The adjusted reference points are defined as:

        .. math::

        { m_i - 10 \mid 0 < i \le n \}

        { M_i + 10 \mid 0 < i \le n \}

        where :math:`0 < i \le n`.

   .. cpp:member:: ReferencePointCalculationStyle BetterReferencePointCalculationStyle
   .. cpp:member:: ReferencePointCalculationStyle WorseReferencePointCalculationStyle

   .. cpp:member:: bool callbacks

      Toggle to enable or disable iteration callbacks.

   .. cpp:member:: int MaxEstimationTime

      Maximum time allowed for the estimation process (in ms).

   .. cpp:member:: Point* worseReferencePoint;

        User defined reference point;for minimization: lower boundary point of the problem; for maximization: upper boundary point of the problem 

   .. cpp:member:: Point* betterReferencePoint;

        User defined reference point; for minimization: upper boundary point of the problem; for maximization: lower boundary point of the problem
        
   .. cpp:function:: Point* GetWorseReferencePoint(DataSet *set)
   .. cpp:function:: Point* GetBetterReferencePoint(DataSet *set)


Derived Solver Parameters
-------------------------

.. cpp:class:: QEHCParameters : public SolverParameters

   Parameters for the QEHC Solver.

   .. cpp:enum:: SearchSubjectOption

      .. cpp:enumerator:: MinimumContribution
      .. cpp:enumerator:: MaximumContribution
      .. cpp:enumerator:: Both

   .. cpp:member:: unsigned long int iterationsLimit

      The maximum number of iterations.

   .. cpp:member:: bool sort

      Flag to allow or disallow sorting.

.. cpp:class:: HSSParameters : public SolverParameters

   Configuration for HSS solvers.

   .. cpp:enum:: StoppingCriteriaType

      .. cpp:enumerator:: SubsetSize
      .. cpp:enumerator:: Time

   .. cpp:enum:: SubsetSelectionStrategy

      .. cpp:enumerator:: Incremental
      .. cpp:enumerator:: Decremental

   .. cpp:member:: StoppingCriteriaType StoppingCriteria
   .. cpp:member:: SubsetSelectionStrategy Strategy

.. cpp:class:: QHV_BQParameters : public SolverParameters

   Settings for QHV-BQ optimization.

   .. cpp:member:: SwitchParameters SwitchToMCSettings

      Configuration for switching to Monte Carlo methods.

   .. cpp:member:: bool MonteCarlo

      Toggle to turn on/off Monte Carlo estimation.

Utility Structures
------------------

.. cpp:class:: SwitchParameters

   Internal configuration for algorithm switching logic.

   .. cpp:member:: int switchTime

      Threshold in ms to switch to Monte Carlo.

   .. cpp:member:: DType gap

      Minimum gap required between reference points to stop estimation.
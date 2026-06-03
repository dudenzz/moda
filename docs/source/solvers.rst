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
        
        For the dataset $X$ set in $R^n$ space, the reference points are defined as $\{\forall_{0 \lt i \le n}\forall_{x \in X}
(min(x_i) - \epsilon)\}$ and $\{\forall_{0 \lt i \le n}\forall_{x \in X}
(max(x_i) + \epsilon)\}$. The $\epsilon$ value is arbitrarily set to 0.001, this value can be adjusted by modyfiing `include.h` file. 

      .. cpp:enumerator:: tenpercent

For the dataset $X$ set in $R^n$ space, the reference points are defined as $\{\forall_{0 \lt i \le n}\forall_{x \in X}
(min(x_i) - 0.1 \cdot \lvert min(x_i) - max(x_i) \rvert )\}$ and $\{\forall_{0 \lt i \le n}\forall_{x \in X}
(max(x_i) + 0.1 \cdot \lvert min(x_i) - max(x_i) \rvert )\}$.

      .. cpp:enumerator:: zeroone

For the dataset $X$ set in $R^n$ space, the reference points are defined as $\{\forall_{x \in X}
(0)\}$ and $\{\forall_{x \in X}
(0)\}$.

      .. cpp:enumerator:: userdefined
The reference points are defined by user prior to the solver execution. In order to use this setting declare the worseReferencePoint and betterReferencePoint values.
      .. cpp:enumerator:: exact
For the dataset :math:`X \subseteq \mathbb{R}^n`, the reference points are defined as:

.. math::

   \text{min\_ref} = \{ \min_{x \in X} (x_i) \mid 0 < i \le n \}

   \text{max\_ref} = \{ \max_{x \in X} (x_i) \mid 0 < i \le n \}
      .. cpp:enumerator:: pymoo
For the dataset $X$ set in $R^n$ space, the reference points are defined as $\{\forall_{0 \lt i \le n}\forall_{x \in X}
(min(x_i) - 10)\}$ and $\{\forall_{0 \lt i \le n}\forall_{x \in X}
(max(x_i) + 10)\}$.

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
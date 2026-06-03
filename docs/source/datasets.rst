DataSet management
=====

.. _tutorials_datasets:

.. cpp:namespace:: moda

DataSet Class
=============

.. cpp:class:: DataSet : public SolutionContainer

   This class represents an optimization problem (a dataset), which is comprised of a set of points.

   .. cpp:enum:: OptimizationType

      .. cpp:enumerator:: minimization
      .. cpp:enumerator:: maximization

      Defines the optimization goal. Defaults to :cpp:enumerator:`maximization`.

   .. rubric:: Public Members

   .. cpp:member:: std::vector<Point*> points

      A set of points in the dataset. 

   .. cpp:member:: std::string filename

      Full path to the source file.

   .. cpp:member:: OptimizationType typeOfOptimization

      Type of this problem.

   .. rubric:: Constructors

   .. cpp:function:: DataSet(int nObjectives = 2)

      Non-parametrized constructor.

   .. cpp:function:: DataSet(const DataSet& dataset)

      Copy constructor.

   .. cpp:function:: DataSet(const std::string filename, bool normalizedName)

      Reads data from a file. If `normalizedName` is true, properties are inferred from the filename 
      (e.g., ``name_of_experiment_dXXX_nXXX_ZZ``).

   .. cpp:function:: DataSet(const std::string filename, DataSetParameters settings)

      Reads data from a file using provided dataset properties.

   .. cpp:function:: DataSet(const std::string filename, std::string name, int dimensions, int sample, int nPoints)

      Reads data from a file with explicitly provided experiment metadata.

   .. rubric:: Accessors & Modifiers

   .. cpp:function:: Point* getIdeal()
   .. cpp:function:: void setIdeal(Point*)
   .. cpp:function:: Point* getNadir()
   .. cpp:function:: void setNadir(Point*)

      Getters and setters for the ideal and nadir points.

   .. cpp:function:: void setParameters(DataSetParameters settings)
   .. cpp:function:: void setDimensionality(int dim)
   .. cpp:function:: void setName(std::string name)
   .. cpp:function:: void setNumberOfPoints(int npts)
   .. cpp:function:: void setSampleNumber(int sampleN)

      Setters for dataset metadata.

   .. rubric:: Stream & File Operations

   .. cpp:function:: std::istream& Load(std::istream& stream)

      Reads the dataset from an input stream.

   .. cpp:function:: void Save(const std::string filename)

      Saves the dataset to a file.

   .. cpp:function:: static DataSet* LoadFromFilename(const std::string filename)

      Factory method to load a dataset from a file.

   .. cpp:function:: static std::vector<DataSet*> LoadBulk(const std::string directory)

      Loads multiple datasets from a directory.

   .. cpp:function:: NDTree<Point> toNDTree()

      Converts the dataset to an NDTree.

   .. rubric:: Functions & Normalization

   .. cpp:function:: void normalize()

      Normalizes the dataset.

   .. cpp:function:: void reverseObjectives()

      Reverses the objectives for the points.

   .. cpp:function:: bool add(Point* point)
   .. cpp:function:: bool remove(Point* point)
   .. cpp:function:: void clear()
   .. cpp:function:: void RemoveDominated()

      Functions to manipulate the collection of points.

   .. rubric:: Grouping

   .. cpp:function:: static std::vector<std::vector<DataSet>> BulkGroup(std::vector<DataSet> problems, ProblemGrouping grouping)

      Groups a vector of datasets according to a given criteria.
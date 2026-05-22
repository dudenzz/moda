Installation on MacOS
=====

.. _installation:

Compiling the files on local machine
------------

Create a working directory. Within this directory create a virtual environment.

``python -m venv venv`` (Windows)

or 

``python3 -m venv venv`` (Linux/MacOS)

Activate the environment. 

``./bin/Scripts/activate`` (Windows)

or

``source venv/bin/activate``(Linux/MacOS)

Clone the MODA repository (main branch).

``git clone https://github.com/dudenzz/moda``

Navigate to the python wrapper source directory.

``cd moda/moda/python-wrap``.

Install the library. As the library uses C-API make sure you are forcing the reinstallation with new binary files created for every installation instance.

``pip install . --force-reinstall --no-binaries moda``

Congratulations! You have successfully installed python wrapper for MODA library in your current virtual environment.


Installation with pip
----------------

Work in progress


Basic usage 
----------------

In order to test the installed library, run the following Python script within the virtual environment:

.. code-block:: python

    import numpy as np
    import moda
    from moda import IQHVParameters, IQHVSolver

    data = np.random.random((10,2))
    solver = IQHVSolver()
    params = IQHVParameters()
    params.WorseReferencePointCalculationStyle = moda.ReferencePointCalculationStyle.tenpercent
    params.BetterReferencePointCalculationStyle = moda.ReferencePointCalculationStyle.tenpercent
    ds = moda.DataSet(data)
    ds.typeOfOptimization = moda.OptimizationType.maximization
    r = solver.Solve(ds,params)
    print(r)




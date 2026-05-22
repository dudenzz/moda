Installation on Windows
=====

.. _installation_win:

Visual Studio
------------

If you are working with Visual Studio IDE, this is the first tutorial you should go through. It explains how to install and configure your MODA library instance within the Visual Studio environment.

Installation
~~~~~~~~~~~~

First you must download the MODA SDK from the :ref:`downloads` page. Note that the Release package allows you to run the code in both Debug and Release mode, while the Debug mode allows you to run the code in Debug mode only, but allows for detailed debugging. Extract the files anywhere you like. Copying the files into your Visual Studio installation folder is not recommended. Choose a dedicated location, especially if you intend to use several versions of the same library or you intend to use various compilers.

Creating the MODA Visual Studio project. 
~~~~~~~~~~~~~

Use the Visual Studio IDE to create a C++ based project. Any type of the project is allowed, however it is recommended to use the *Empty Application*.
For the purpose of this tutorial, you should create a main.cpp file and add it to the project, so that we have access to the C++ settings (otherwise Visual Studio doesn't know which language you're going to use for this project).

First we need to configure the IDE to use the proper standard. MODA library is tested and compatible with C++17 and later versions. We recommend using the latest C++ version available.
In the project properties under "C/C++" Language tab, change the "C++ Language Standard" dropdown to the desired version (C++17 or higher). 

.. figure:: images/C++version.png
   :width: 600
   :align: center

   Pick the proper C++ version. 

Now we need to tell the compiler where to find the SFML headers (.hpp files), and the linker where to find the SFML libraries (.lib files).

In the project's properties, add:

The path to the MODA headers (<moda-install-path>/moda) to C/C++ > General > Additional Include Directories
The path to the MODA libraries (<moda-install-path>) to Linker > General > Additional Library Directories
.. figure:: images/include_directories.png
   :width: 600
   :align: center

   Headers directory. 

.. figure:: images/lib_directories.png
   :width: 600
   :align: center

   Library directory. 


In the next step we need to tell the compiler, that it is supposed to use the `moda.lib` library. Head to  Linker > Input > Additional Dependencies. Add all the MODA library: "moda.lib".



.. figure:: images/lib_name.png
   :width: 600
   :align: center

   Library directory. 

Your project is ready, let's write some code now to make sure that it works. Put the following code inside the main.cpp file:

.. code-block:: cpp

   #include <moda\DataSet.h>
   #include <moda\IQHVSolver.h>
   #include <moda\SolverParameters.h>
   #include <moda\Point.h>
   #include <iostream>
   int main()
   {
      moda::DataSet* dataSet = new moda::DataSet(2);
      for (int i = 0; i < 10; i++)
      {
         moda::Point* newPoint = new moda::Point(2);
         newPoint->ObjectiveValues[0] = i * 0.1;
         newPoint->ObjectiveValues[1] = i * 0.1;
         dataSet->add(newPoint);
      }
      moda::IQHVSolver solver;
      moda::IQHVParameters* parameters = new moda::IQHVParameters(moda::SolverParameters::ReferencePointCalculationStyle::zeroone, moda::SolverParameters::ReferencePointCalculationStyle::zeroone);
      auto result = solver.Solve(dataSet, *parameters);
      std::cout << "Hypervolume: " << result->HyperVolume << std::endl;
      return 0;
   }

If you recieve the following output: ``Hypervolume: 0.81``, the library has been installed properly.

CMake
----------------

Work in progress

VCPKG
----------------

Work in progress





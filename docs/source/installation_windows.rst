Installation on Windows
=====

.. _installation_win:

Visual Studio
------------

If you are working with Visual Studio IDE, this is the first tutorial you should go through. It explains how to install and configure your MODA library instance within the Visual Studio environment.

Installation
~~~~~~~~~~~~

First you must download the MODA SDK from the :ref:`downloads` page. Extract the files anywhere you like. Copying the files into your Visual Studio installation folder is not recommended. Choose a dedicated location, especially if you intend to use several versions of the same library or you intend to use various compilers.

Creating the MODA Visual Studio project. 
~~~~~~~~~~~~~

Use the Visual Studio IDE to create a C++ based project. Any type of the project is allowed, however it is recommended to use the *Empty Application*.

First we need to configure the IDE to use the proper standard. MODA library is tested and compatible with C++17 and later versions. We recommend using the latest C++ version available.
In the project properties under "C/C++" Language tab, change the "C++ Language Standard" dropdown to the desired version (C++17 or higher). 

.. figure:: docs/images/C++version.png
   :width: 600
   :align: center

   Pick the proper C++ version. 

CMake
----------------

Work in progress

VCPKG
----------------

Work in progress





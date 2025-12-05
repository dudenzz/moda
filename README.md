# MODA
A comprehensive suite of hypervolume-related algorithms designed for efficient multi-objective optimization

## Installation guide
The library is going to be abailiable via packet managers PyPi (python wrap) and VCPKG (C++ original library). Additionally, the library makes use of the CMake for Windows and the Make for Linux based operating systems. 

Currently the library is under pull request review for VCPKG, hence we provide an ad-hoc installation guideline. Note that this is work in progress, and this process should be used with necessary care.

1. Clone (or download and unzip) the [VCPKG](https://github.com/microsoft/vcpkg) repository.
2. Run `bootstrap-vcpkg.bat`
   (optionally) Set up environmental variables for the VCPKG
3. Create a new directory in `vcpkg/ports` and name it `moda`
4. From this repository, the `vcpkg-moda` directory, download the `portfile.cmake` and `vcpkg.json`. Paste those files into the newly created `vcpkg/ports/moda` directory.
5. Direct to main vcpkg directory and install moda library with the `.\vcpkg.exe install moda` command (by default it installs for x64 architecture, if you wish to use any other architecture, use additional triplet, such as `.\vcpkg.exe install moda:x86-windows-static`)
6. Run the `.\vcpkg.exe integrate install`

The package is now installed. If you are using a CMake based project, you are good to go. If however you are using a Visual Studio project, there is one more thing to do. Find the desired comiplation of moda library in `packages` directory in VCPKG. Within that directory navigate to `share` and copy the path of `moda.lib` file in that directory. In Visual Studio open the Project Properties. Navigate to Linker and append the copied path at the end of Additional Dependencies.

### Minimal program

After the installation, you should be able to include the library files in any project. 

In the following example, I am loading the dataset from file (sample included in repository). Then I'm displaying the data points one by one. In this library, the algorithmic functionalities are provided via `Solvers`. Here I'm using the `IQHVSolver`. This solver is dedicated for calculating the Hypervolume of a given dataset. The solvers are parametrized with the parameters object, each solver has its own Parameters class - in this case I make use of `IQHVParameters`. This is the simpliest parameters class and it is limitted to calculation formulas for reference points. I run the Solve function, which returns a result, which is comprised of the elapsed time and calculated hypervolume. I display the calculated hypervolume.
```cpp

#include <iostream>
#include "moda\DataSet.h"
#include "moda\IQHVSolver.h"
#include "moda\Result.h"
int main()
{
    //load data
    moda::DataSet* ds = moda::DataSet::LoadFromFilename("data_6_500_convex_triangular_1"); 
    //display data
    for (auto p : ds->points)
    {
        for(int i = 0; i<6;i++) std::cout << p->ObjectiveValues[i] << " ";
        std::cout << std::endl;
    }
    
    ds->typeOfOptimization = moda::DataSet::minimization; //parametrize data
    ds->normalize(); //manipulate data
    moda::IQHVSolver solver; //create solver
    moda::IQHVParameters params; //create solver parameters
    //set up parameters
    params.BetterReferencePointCalculationStyle = moda::IQHVParameters::ReferencePointCalculationStyle::zeroone;
    params.WorseReferencePointCalculationStyle = moda::IQHVParameters::ReferencePointCalculationStyle::zeroone;
    //calculate hypervolume
    moda::HypervolumeResult * result = solver.Solve(ds, params);
    //display result
    std::cout << result->HyperVolume;
}
```
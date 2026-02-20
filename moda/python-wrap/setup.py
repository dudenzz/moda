# setup.py (Example for setuptools)

from setuptools import setup, Extension
import numpy as np
import sys

# Define the C/C++ extension module

if sys.platform == 'win32':
    # MSVC Flags
    compile_args = ['/std:c++17', '/Zi'] # /Zi adds debug info
    link_args = ['/DEBUG']
else:
    # GCC/Clang Flags (Linux/WSL)
    compile_args = ['-std=c++17', '-fsanitize=address', '-fno-omit-frame-pointer', '-g']
    link_args = ['-fsanitize=address']
    
modamodule = Extension(
    'moda',  # The name of the compiled module
    sources=['moda.cpp', 
             'py_point.cpp', '../Point.cpp',
             'py_dataset.cpp', '../DataSet.cpp',
              '../NDTree.cpp',
              '../TreeNode.cpp',
              '../DataSetParameters.cpp',
              '../ListSet.cpp',
              '../Helpers.cpp',
              '../myvector.cpp',
              '../Result.cpp',
              '../ExecutionPool.cpp', '../ExecutionContext.cpp', '../ExecutionService.cpp', '../DynamicStructures.cpp',
              '../SubProblemsStackLevel.cpp', '../SubproblemsPool.cpp', '../SubproblemsStackPriorityQueue.cpp',
              'py_solver_parameters.cpp','../SolverParameters.cpp',
              'py_solver.cpp', '../Solver.cpp',
              'py_qehc_solver.cpp', '../QEHCSolver.cpp',
              'py_iqhv_solver.cpp', '../IQHVSolver.cpp',
              'py_hss_solver.cpp', '../HSSSolver.cpp',
              '../QEHC.cpp', '../Hypervolume.cpp', '../IQHV.cpp', '../HSS.cpp','../ObjectivesTransformer.cpp'
             
            ], 
    include_dirs=[np.get_include(), '.'], # Include numpy headers and local headers
    language='c++',
    extra_compile_args=compile_args, # Use modern C++ standard
    extra_link_args=link_args
)

setup(
    name='moda',
    version='1.0.0',
    packages=['moda'], # This creates the moda package folder/structure
    ext_modules=[modamodule],
    install_requires=[
        'numpy>=1.18.0',
    ],
    # place moda.py inside a 'moda' directory.
)
# setup.py (Example for setuptools)

from setuptools import setup, Extension
import numpy as np
import sys

# Define the C/C++ extension module

if sys.platform == 'win32':
    # Flagi dla kompilatora MSVC (Windows)
    compile_args = ['/std:c++17']
else:
    # Flagi dla GCC/Clang (Linux/macOS)
    compile_args = ['-std=c++17']
    
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
              'py_solver_parameters.cpp','../SolverParameters.cpp',
              'py_solver.cpp', '../Solver.cpp',
              'py_qehc_solver.cpp', '../QEHCSolver.cpp'
             
            ], 
    include_dirs=[np.get_include(), '.'], # Include numpy headers and local headers
    language='c++',
    extra_compile_args=compile_args, # Use modern C++ standard
)

setup(
    name='moda',
    version='1.0.0',
    packages=['moda'], # This creates the moda package folder/structure
    ext_modules=[modamodule],
    install_requires=[
        'numpy>=1.18.0',
    ],
    # The Python module is not explicitly needed here if using a simple structure, 
    # but in a real project, you would place moda.py inside a 'moda' directory.
)
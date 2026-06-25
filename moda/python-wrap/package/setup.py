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
    # compile_args = ['-std=c++17', '-fsanitize=address', '-fno-omit-frame-pointer', '-g']
    compile_args = ['-std=c++17']
    # link_args = ['-fsanitize=address']
    link_args = []
    
modamodule = Extension(
    'moda',  # The name of the compiled module
    sources=['moda.cpp', 
             'py_point.cpp', '../../Point.cpp',
             'py_dataset.cpp', '../../DataSet.cpp',
              '../../NDTree.cpp',
              '../../TreeNode.cpp',
              '../../DataSetParameters.cpp',
              '../../ListSet.cpp',
              '../../Helpers.cpp',
              '../../myvector.cpp',
              '../../Result.cpp',
              '../../ExecutionPool.cpp', '../../ExecutionContext.cpp', '../../ExecutionService.cpp', '../../DynamicStructures.cpp',
              '../../SubProblemsStackLevel.cpp', '../../SubproblemsPool.cpp', '../../SubproblemsStackPriorityQueue.cpp',
              'py_solver_parameters.cpp','../../SolverParameters.cpp',
              'py_solver.cpp', '../../Solver.cpp',
              'py_qehc_solver.cpp', '../../QEHCSolver.cpp',
              'py_iqhv_solver.cpp', '../../IQHVSolver.cpp',
              'py_hss_solver.cpp', '../../HSSSolver.cpp',
              '../../QEHC.cpp', '../../Hypervolume.cpp', '../../IQHV.cpp', '../../HSS.cpp','../../ObjectivesTransformer.cpp'
             
            ], 
    include_dirs=[np.get_include(), '.'], # Include numpy headers and local headers
    language='c++',
    extra_compile_args=compile_args # Use modern C++ standard
    # extra_link_args=link_args
)

setup(
name='put-moda',
    version='1.0.21',
    author='Jakub Dutkiewicz',
    author_email='jakub.dutkiewicz@put.poznan.pl',
    description='Multiobjective Optimization Data structures and Algorithms (MODA) - Python bindings for C++ library',
    long_description='MODA is a comprehensive suite of hypervolume-related algorithms designed for efficient multi-objective optimization.',
    long_description_content_type='text/markdown',
    url='https://github.com/dudenzz/moda', 
    ext_modules=[modamodule],
    install_requires=['numpy>=1.18.0'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.3',)
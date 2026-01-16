#pragma once

#include <Python.h>
#include "../Solver.h" // Actual C++ Solver header
#include "../Point.h"
#include "../myvector.h"

// Base Solver
typedef struct {
    PyObject_HEAD
    moda::Solver *solver;
    // callbacks
    PyObject *start_callback; 
    PyObject *iteration_callback;
    PyObject *end_callback;
} SolverObject;

//  Derived solvers
typedef struct {
    SolverObject super; 
    moda::myvector<moda::Point*> indexSet;
} QEHCSolverObject;

typedef struct {
    SolverObject super; 
} IQHVSolverObject;

typedef struct {
    SolverObject super; 
} HSSSolverObject;

// Data containers
typedef struct {
    PyObject_HEAD
    moda::DataSet *data_set;
} DataSetObject;

// Structure for the Python Point object
typedef struct {
    PyObject_HEAD             // Required boilerplate for Python objects
    moda::Point *point;       // Pointer to the actual C++ Point object
} PointObject;

// Parameters
typedef struct {
    PyObject_HEAD
    moda::SolverParameters *params;
} SolverParametersObject;

typedef struct {
    SolverParametersObject base; 
    moda::QEHCParameters *innerParams;
} QEHCParametersObject;

typedef struct {
    SolverParametersObject base; 
    moda::IQHVParameters *innerParams;
} IQHVParametersObject;

typedef struct {
    SolverParametersObject base; 
    moda::HSSParameters *innerParams;
} HSSParametersObject;

// 4. Declare the Type objects as extern so other files can see them
#ifdef __cplusplus
extern "C" {
#endif

// 1. Declare the Types
extern PyTypeObject SolverType;
extern PyTypeObject QEHCSolverType;
extern PyTypeObject IQHVSolverType;
extern PyTypeObject HSSSolverType;
extern PyTypeObject DataSetType;
extern PyTypeObject PointType;
extern PyTypeObject QEHCParametersType;
extern PyTypeObject IQHVParametersType;
extern PyTypeObject HSSParametersType;
extern PyTypeObject SolverParametersType;
extern PyTypeObject ReferencePointCalculationStyleType;
extern PyTypeObject SearchSubjectOptionType;
extern PyTypeObject StoppingCriteriaTypeType;
extern PyTypeObject SubsetSelectionStrategyType;
extern PyTypeObject OptimizationTypeType;

// 2. Declare the Helper Functions
int init_SearchSubjectOption(PyObject *m);
int init_ReferencePointCalculationStyle(PyObject *m);
int init_SubsetSelectionStrategy(PyObject *m);
int init_StoppingCriteriaType(PyObject *m);
int init_OptimizationType(PyObject *m);


PyObject* Point_create_copy(moda::Point* cpp_point);

#ifdef __cplusplus
}
#endif
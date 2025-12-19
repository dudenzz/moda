#pragma once

#include <Python.h>
#include "../Solver.h" // Actual C++ Solver header
#include "../Point.h"
#include "../myvector.h"

// Struktura dla Solver
typedef struct {
    PyObject_HEAD
    moda::Solver *solver;
    // Potrzebujemy też miejsca na przechowywanie referencji do obiektów Callbacks
    PyObject *start_callback; 
    PyObject *iteration_callback;
    PyObject *end_callback;
} SolverObject;

// 2. Define the derived structure
typedef struct {
    SolverObject super; 
    moda::myvector<moda::Point*> indexSet;
} QEHCSolverObject;

// 3. Define the DataSet structure
typedef struct {
    PyObject_HEAD
    moda::DataSet *data_set;
} DataSetObject;

// Structure for the Python Point object
typedef struct {
    PyObject_HEAD             // Required boilerplate for Python objects
    moda::Point *point;       // Pointer to the actual C++ Point object
} PointObject;

typedef struct {
    PyObject_HEAD
    moda::SolverParameters *params;
} SolverParametersObject;

typedef struct {
    SolverParametersObject base; // Musi być PIERWSZE dla dziedziczenia w Pythonie!
    moda::QEHCParameters *innerParams;
} QEHCParametersObject;

// 4. Declare the Type objects as extern so other files can see them
#ifdef __cplusplus
extern "C" {
#endif

// 1. Declare the Types
extern PyTypeObject SolverType;
extern PyTypeObject QEHCSolverType;
extern PyTypeObject DataSetType;
extern PyTypeObject PointType;
extern PyTypeObject QEHCParametersType;
extern PyTypeObject SolverParametersType;
extern PyTypeObject ReferencePointCalculationStyleType;
extern PyTypeObject SearchSubjectOptionType;
extern PyTypeObject OptimizationTypeType;

// 2. Declare the Helper Functions
int init_SearchSubjectOption(PyObject *m);
int init_ReferencePointCalculationStyle(PyObject *m);
int init_OptimizationType(PyObject *m);

PyObject* Point_create_copy(moda::Point* cpp_point);

#ifdef __cplusplus
}
#endif
#pragma once
#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL moda_ARRAY_API
#include <numpy/arrayobject.h> 
#include "moda_types.h"


// Forward declarations
extern PyTypeObject PointType; 
extern PyTypeObject DatasetType; 
extern PyTypeObject SolverType; 
extern PyTypeObject QEHCSolverType;
extern PyTypeObject IQHVSolverType;
extern PyTypeObject HSSSolverType;
extern PyTypeObject ReferencePointCalculationStyleType;
extern PyTypeObject SearchSubjectOptionType;
extern PyTypeObject SubsetSelectionStrategyType;
extern PyTypeObject StoppingCriteriaTypeType;
extern PyTypeObject QEHCParametersType;
extern PyTypeObject IQHVParametersType;
extern PyTypeObject HSSParametersType;
extern PyTypeObject SolverParametersType;
// --- Module Methods ---
static PyMethodDef ModaMethods[] = {
    {NULL, NULL, 0, NULL}  // Sentinel
};

// --- Module Definition Structure ---
static struct PyModuleDef modamodule = {
    PyModuleDef_HEAD_INIT,
    "_moda", // Name of the module (must match the compiled file name)
    "Python wrapper for the moda C++ library.", // Module documentation
    -1, // Size of per-interpreter state of the module, or -1 if the module keeps state in global variables.
    ModaMethods // Methods table
};

// --- Module Initialization Function ---
PyMODINIT_FUNC
PyInit_moda(void)
{
    PyObject *m;

    // IMPORTANT: Initialize NumPy C-API
    import_array(); 

    // 1. Finalize the types
    if (PyType_Ready(&PointType) < 0)
        return NULL;
    if (PyType_Ready(&DataSetType) < 0)
        return NULL;
    if (PyType_Ready(&SolverParametersType) < 0)
        return NULL;
    if (PyType_Ready(&QEHCParametersType) < 0)
        return NULL;
    if (PyType_Ready(&IQHVParametersType) < 0)
        return NULL;    
    if (PyType_Ready(&HSSParametersType) < 0)
        return NULL;
    if (PyType_Ready(&ReferencePointCalculationStyleType) < 0)
        return NULL;
    if (PyType_Ready(&SearchSubjectOptionType) < 0)
        return NULL;
    if (PyType_Ready(&SubsetSelectionStrategyType) < 0)
        return NULL;
    if (PyType_Ready(&StoppingCriteriaTypeType) < 0)
        return NULL;
    if (PyType_Ready(&SolverType) < 0) 
        return NULL;
    if (PyType_Ready(&QEHCSolverType) < 0) 
        return NULL;
    if (PyType_Ready(&IQHVSolverType) < 0) 
        return NULL;
    if (PyType_Ready(&HSSSolverType) < 0) 
        return NULL;


    // 2. Create the module object
    m = PyModule_Create(&modamodule);
    if (m == NULL)
        return NULL;

    // 3. Add the types to the module
    Py_INCREF(&PointType);
    if (PyModule_AddObject(m, "Point", (PyObject *)&PointType) < 0) {
        Py_DECREF(&PointType);
        Py_DECREF(m);
        return NULL;
    }

    Py_INCREF(&DataSetType);
    if (PyModule_AddObject(m, "DataSet", (PyObject *)&DataSetType) < 0) {
        Py_DECREF(&DataSetType);
        Py_DECREF(m);
        return NULL;
    }

    Py_INCREF(&SolverParametersType);
    if (PyModule_AddObject(m, "SolverParameters", (PyObject *)&SolverParametersType) < 0) {
        Py_DECREF(&SolverParametersType);
        Py_DECREF(m);
        return NULL;
    }

    Py_INCREF(&QEHCParametersType);
    if (PyModule_AddObject(m, "QEHCParameters", (PyObject *)&QEHCParametersType) < 0) {
        Py_DECREF(&QEHCParametersType);
        Py_DECREF(m);
        return NULL;
    }
    Py_INCREF(&IQHVParametersType);
    if (PyModule_AddObject(m, "IQHVParameters", (PyObject *)&IQHVParametersType) < 0) {
        Py_DECREF(&IQHVParametersType);
        Py_DECREF(m);
        return NULL;
    }
    Py_INCREF(&HSSParametersType);
    if (PyModule_AddObject(m, "HSSParameters", (PyObject *)&HSSParametersType) < 0) {
        Py_DECREF(&HSSParametersType);
        Py_DECREF(m);
        return NULL;
    }
    Py_INCREF(&SolverType);
    if (PyModule_AddObject(m, "Solver", (PyObject *)&SolverType) < 0) {
        Py_DECREF(&SolverType);
        Py_DECREF(m);
        return NULL;
    }
    Py_INCREF(&QEHCSolverType);
    if (PyModule_AddObject(m, "QEHCSolver", (PyObject *)&QEHCSolverType) < 0) {
        Py_DECREF(&QEHCSolverType); Py_DECREF(m); return NULL;
    }
        Py_INCREF(&IQHVSolverType);
    if (PyModule_AddObject(m, "IQHVSolver", (PyObject *)&IQHVSolverType) < 0) {
        Py_DECREF(&IQHVSolverType); Py_DECREF(m); return NULL;
    }

        if (PyModule_AddObject(m, "HSSSolver", (PyObject *)&HSSSolverType) < 0) {
        Py_DECREF(&HSSSolverType); Py_DECREF(m); return NULL;
    }
    if (init_ReferencePointCalculationStyle(m) < 0) return NULL;
    if (init_SearchSubjectOption(m) < 0) return NULL;
    if (init_OptimizationType(m) < 0) return NULL;
    if (init_SubsetSelectionStrategy(m) < 0) return NULL;
    if (init_StoppingCriteriaType(m) < 0) return NULL;
    return m;
}
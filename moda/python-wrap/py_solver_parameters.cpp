#pragma once
#include <Python.h>
#include "../SolverParameters.h"
#include <structmember.h>
#include <numpy/arrayobject.h>
#include "moda_types.h"

#pragma region enums
//Enum type definition for reference points calculation style
PyTypeObject ReferencePointCalculationStyleType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "moda.ReferencePointCalculationStyle",
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 17 zer dla tp_basicsize do tp_iternext
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,                                   // tp_flags
    "Reference points calculation style"
};



// Enum type definition for search subject options
PyTypeObject SearchSubjectOptionType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "moda.QEHCParameters.SearchSubjectOption",
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, 
    "Search Subject (minimum contribution, maximum contribution).",
};


// Enum type definition for subset selection strategy
PyTypeObject SubsetSelectionStrategyType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "moda.HSSParameters.SubsetSelectionStrategy",
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, 
    "Subset Selection Strategy for HSS algorithms (decremental or incremental)",
};

// Enum type definition for subset selection strategy
PyTypeObject StoppingCriteriaTypeType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "moda.HSSParameters.StoppingCriteriaType",
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, 
    "Subset Selection Stopping Criteria for HSS algorithms (time or size)",
};

//Enum initializations
int init_ReferencePointCalculationStyle(PyObject *m) {
    PyObject *enum_type_obj = (PyObject *)&ReferencePointCalculationStyleType;

    // 1. We put the enum into the module
    Py_INCREF(enum_type_obj);
    if (PyModule_AddObject(m, "ReferencePointCalculationStyle", enum_type_obj) < 0) {
        Py_DECREF(enum_type_obj);
        return -1;
    }
    
    // 2. Dictionary acquisition
    PyObject *dict = ((PyTypeObject *)enum_type_obj)->tp_dict;
    if (!dict) {
        PyErr_SetString(PyExc_RuntimeError, "Type dictionary is NULL.");
        return -1;
    }

    // Macro for filling the dictionary
    #define ADD_ENUM_CONST_TO_DICT(name, val) \
        do { \
            PyObject *v = PyLong_FromLong(val); \
            if (!v) return -1; \
            int result = PyDict_SetItemString(dict, name, v); \
            Py_DECREF(v); \
            if (result < 0) return -1; \
        } while (0)

    // Adding literals to the dictionary
    ADD_ENUM_CONST_TO_DICT("epsilon", moda::SolverParameters::epsilon);
    ADD_ENUM_CONST_TO_DICT("tenpercent", moda::SolverParameters::tenpercent);
    ADD_ENUM_CONST_TO_DICT("zeroone", moda::SolverParameters::zeroone);
    ADD_ENUM_CONST_TO_DICT("userdefined", moda::SolverParameters::userdefined);
    ADD_ENUM_CONST_TO_DICT("exact", moda::SolverParameters::exact);

    #undef ADD_ENUM_CONST_TO_DICT
    
    return 0;
}
//Enum initializations
int init_SearchSubjectOption(PyObject *m) {

    PyObject *enum_type_obj = (PyObject *)&SearchSubjectOptionType;
    
    // 
    if (PyType_Ready(&SearchSubjectOptionType) < 0) return -1;
    
    // 1. We put the enum into the module
    Py_INCREF(enum_type_obj);
    if (PyModule_AddObject(m, "SearchSubjectOption", enum_type_obj) < 0) {
        Py_DECREF(enum_type_obj);
        return -1;
    }

    
    // 2. Dictionary acquisition
    PyObject *dict = ((PyTypeObject *)enum_type_obj)->tp_dict;
    if (!dict) {
        PyErr_SetString(PyExc_RuntimeError, "Type dictionary is NULL for SearchSubjectOptionType.");
        return -1;
    }

    // Macro for filling the dictionary
    #define ADD_ENUM_CONST_TO_DICT(name, val) \
        do { \
            PyObject *v = PyLong_FromLong(val); \
            if (!v) return -1; \
            int result = PyDict_SetItemString(dict, name, v); \
            Py_DECREF(v); \
            if (result < 0) return -1; \
        } while (0)

    // 4. Adding literals to the dictionary
    ADD_ENUM_CONST_TO_DICT("MinimumContribution", moda::QEHCParameters::MinimumContribution);
    ADD_ENUM_CONST_TO_DICT("MaximumContribution", moda::QEHCParameters::MaximumContribution);
    ADD_ENUM_CONST_TO_DICT("Both", moda::QEHCParameters::Both);
    
    #undef ADD_ENUM_CONST_TO_DICT
    
    return 0;
}

//Enum initializations
int init_StoppingCriteriaType(PyObject *m) {
    // Obiekt typu, do którego dodajemy stałe
    PyObject *enum_type_obj = (PyObject *)&StoppingCriteriaTypeType;
    
    // 
    if (PyType_Ready(&StoppingCriteriaTypeType) < 0) return -1;
    
    // 1. We put the enum into the module
    Py_INCREF(enum_type_obj);
    if (PyModule_AddObject(m, "StoppingCriteriaType", enum_type_obj) < 0) {
        Py_DECREF(enum_type_obj);
        return -1;
    }

    
    // 2. Dictionary acquisition
    PyObject *dict = ((PyTypeObject *)enum_type_obj)->tp_dict;
    if (!dict) {
        PyErr_SetString(PyExc_RuntimeError, "Type dictionary is NULL for StoppingCriteriaTypeType.");
        return -1;
    }

    // Macro for filling the dictionary
    #define ADD_ENUM_CONST_TO_DICT(name, val) \
        do { \
            PyObject *v = PyLong_FromLong(val); \
            if (!v) return -1; \
            int result = PyDict_SetItemString(dict, name, v); \
            Py_DECREF(v); \
            if (result < 0) return -1; \
        } while (0)

    // 4. Adding literals to the dictionary
    ADD_ENUM_CONST_TO_DICT("SubsetSize", moda::HSSParameters::SubsetSize);
    ADD_ENUM_CONST_TO_DICT("Time", moda::HSSParameters::Time);

    
    #undef ADD_ENUM_CONST_TO_DICT
    
    return 0;
}

//Enum initializations
int init_SubsetSelectionStrategy(PyObject *m) {
    // Obiekt typu, do którego dodajemy stałe
    PyObject *enum_type_obj = (PyObject *)&SubsetSelectionStrategyType;
    
    // 
    if (PyType_Ready(&SubsetSelectionStrategyType) < 0) return -1;
    
    // 1. We put the enum into the module
    Py_INCREF(enum_type_obj);
    if (PyModule_AddObject(m, "SubsetSelectionStrategy", enum_type_obj) < 0) {
        Py_DECREF(enum_type_obj);
        return -1;
    }

    
    // 2. Dictionary acquisition
    PyObject *dict = ((PyTypeObject *)enum_type_obj)->tp_dict;
    if (!dict) {
        PyErr_SetString(PyExc_RuntimeError, "Type dictionary is NULL for SubsetSelectionStrategyType.");
        return -1;
    }

    // Macro for filling the dictionary
    #define ADD_ENUM_CONST_TO_DICT(name, val) \
        do { \
            PyObject *v = PyLong_FromLong(val); \
            if (!v) return -1; \
            int result = PyDict_SetItemString(dict, name, v); \
            Py_DECREF(v); \
            if (result < 0) return -1; \
        } while (0)

    // 4. Adding literals to the dictionary
    ADD_ENUM_CONST_TO_DICT("Incremental", moda::HSSParameters::MinimumContribution);
    ADD_ENUM_CONST_TO_DICT("Decremental", moda::HSSParameters::MaximumContribution);

    
    #undef ADD_ENUM_CONST_TO_DICT
    
    return 0;
}
#pragma endregion
#pragma region SolverParameters
void SolverParameters_dealloc(SolverParametersObject *self) {
    if (self->params) {
        delete self->params;
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

int SolverParameters_init(SolverParametersObject *self, PyObject *args, PyObject *kwds) {
    // Parser argumentów: wszystkie opcjonalne, na podstawie domyślnego konstruktora C++
    int worseStyle = moda::SolverParameters::epsilon;
    int betterStyle = moda::SolverParameters::epsilon;
    int maxTime = 1000;
    int callbacks = 1; // bool w C to int w C-API
    
    if (!PyArg_ParseTuple(args, "|iiip", 
                          &worseStyle, &betterStyle, &maxTime, &callbacks)) {
        return -1;
    }
    
    try {
        // Wywołanie konstruktora C++
        self->params = new moda::SolverParameters(
            (moda::SolverParameters::ReferencePointCalculationStyle)worseStyle,
            (moda::SolverParameters::ReferencePointCalculationStyle)betterStyle,
            maxTime, 
            (bool)callbacks
        );
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return -1;
    }
    
    return 0;
}



PyObject *SolverParameters_get_WorseStyle(SolverParametersObject *self, void *closure) {
    return PyLong_FromLong((long)self->params->WorseReferencePointCalculationStyle);
}

int SolverParameters_set_WorseStyle(SolverParametersObject *self, PyObject *value, void *closure) {
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError, "Cannot delete the WorseReferencePointCalculationStyle attribute");
        return -1;
    }
    long style_val = PyLong_AsLong(value);
    if (style_val < moda::SolverParameters::epsilon || style_val > moda::SolverParameters::exact) {
        PyErr_SetString(PyExc_ValueError, "Invalid value for ReferencePointCalculationStyle.");
        return -1;
    }
    self->params->WorseReferencePointCalculationStyle = (moda::SolverParameters::ReferencePointCalculationStyle)style_val;
    return 0;
}

PyObject *SolverParameters_get_BetterStyle(SolverParametersObject *self, void *closure) {
    return PyLong_FromLong((long)self->params->BetterReferencePointCalculationStyle);
}

int SolverParameters_set_BetterStyle(SolverParametersObject *self, PyObject *value, void *closure) {
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError, "Cannot delete the BetterReferencePointCalculationStyle attribute");
        return -1;
    }
    long style_val = PyLong_AsLong(value);
    if (style_val < moda::SolverParameters::epsilon || style_val > moda::SolverParameters::exact) {
        PyErr_SetString(PyExc_ValueError, "Invalid value for ReferencePointCalculationStyle.");
        return -1;
    }
    self->params->BetterReferencePointCalculationStyle = (moda::SolverParameters::ReferencePointCalculationStyle)style_val;
    return 0;
}


PyObject *SolverParameters_get_betterRefPoint(SolverParametersObject *self, void *closure) {
    if (!self->params->betterReferencePoint) {
        Py_RETURN_NONE;
    }
    // Wywołanie funkcji pomocniczej do opakowania Point* w PointObject
    // Wymaga dostępu do PointObject, która musi być zdefiniowana gdzie indziej (point_wrap.c)
    // Zwracamy KOPIĘ wskaźnika C++ (musisz mieć zaimplementowany konstruktor kopiujący PointObject)
    
    // Uproszczenie: Zwróćmy nowy obiekt PointObject opakowujący KOPIĘ Point*
    extern PyObject* Point_create_copy(moda::Point*); // Zakładana funkcja z point_wrap.c
    
    try {
        return Point_create_copy(self->params->betterReferencePoint);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

PyObject *SolverParameters_get_worseRefPoint(SolverParametersObject *self, void *closure) {
    if (!self->params->betterReferencePoint) {
        Py_RETURN_NONE;
    }
    // Wywołanie funkcji pomocniczej do opakowania Point* w PointObject
    // Wymaga dostępu do PointObject, która musi być zdefiniowana gdzie indziej (point_wrap.c)
    // Zwracamy KOPIĘ wskaźnika C++ (musisz mieć zaimplementowany konstruktor kopiujący PointObject)
    
    // Uproszczenie: Zwróćmy nowy obiekt PointObject opakowujący KOPIĘ Point*
    extern PyObject* Point_create_copy(moda::Point*); // Zakładana funkcja z point_wrap.c
    
    try {
        return Point_create_copy(self->params->worseReferencePoint);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

PyGetSetDef SolverParameters_getsetters[] = {
    {"WorseReferencePointCalculationStyle", 
     (getter)SolverParameters_get_WorseStyle, (setter)SolverParameters_set_WorseStyle,
     "ReferencePointCalculationStyle for the worse reference point.", NULL},

       {"BetterReferencePointCalculationStyle", 
     (getter)SolverParameters_get_BetterStyle, (setter)SolverParameters_set_BetterStyle,
     "ReferencePointCalculationStyle for the worse reference point.", NULL},
    
    {"betterReferencePoint",
     (getter)SolverParameters_get_betterRefPoint, NULL, // Na razie brak settera (złożoność własności)
     "User defined better reference point (Point*). Returns copy.", NULL},
         {"worseReferencePoint",
     (getter)SolverParameters_get_worseRefPoint, NULL, // Na razie brak settera (złożoność własności)
     "User defined better reference point (Point*). Returns copy.", NULL},

    {NULL}  /* Sentinel */
};

// PyMemberDef SolverParameters_members[] = {
//     {"callbacks", 
//      T_BOOL, 
//      offsetof(SolverParametersObject, params) + offsetof(moda::SolverParameters, callbacks), 
//      0,
//      "should the solver use iteration callbacks"},
     
//     {"MaxEstimationTime", 
//      T_INT, 
//      offsetof(SolverParametersObject, params) + offsetof(moda::SolverParameters, MaxEstimationTime), 
//      0,
//      "maximum time of estimation in ms"},
     
//     {"seed", 
//      T_UINT, // unsigned int
//      offsetof(SolverParametersObject, params) + offsetof(moda::SolverParameters, seed), 
//      0,
//      "Custom random seed"},
     
//     {NULL}
// };

PyTypeObject SolverParametersType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "moda.SolverParameters",                    /* tp_name */
    sizeof(SolverParametersObject),             /* tp_basicsize */
    0,                                          /* tp_itemsize */
    (destructor)SolverParameters_dealloc,       /* tp_dealloc */
    0,                                          /* tp_print (Deprecated) */
    0,                                          /* tp_getattr (Deprecated) */
    0,                                          /* tp_setattr (Deprecated) */
    0,                                          /* tp_compare (Deprecated) */
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number: Used for arithmetic operators (+, -, *, etc.) */                              
    0,                                          /* tp_as_sequence: Used for sequence protocols (tuple, list) */ 
    0,                                          /* tp_as_mapping: Used for indexing (p[i]) */ 
    0,                                          /* tp_hash */ 
    0,                                          /* tp_call */ 
    0,                                          /* tp_str: Used for str(p) */
    PyObject_GenericGetAttr,                    /* tp_getattro */
    PyObject_GenericSetAttr,                    /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags: KLUCZOWE dla dziedziczenia */
    "Base class for all solver parameters.",    /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare: For comparisons (<, >, ==) */
    0,                                          /* tp_weaklistoffset */ 
    0,                                          /* tp_iter */ 
    0,                                          /* tp_iternext */ 
    NULL,                                       /* tp_methods (dla GetWorse/BetterReferencePoint) */
    0,                                          /* tp_members */
    SolverParameters_getsetters,                /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    (initproc)SolverParameters_init,            /* tp_init */
    0,                                          /* tp_alloc */
    (newfunc)PyType_GenericNew,                 /* tp_new */
};

#pragma endregion
#pragma region QEHC
int QEHCParameters_init(QEHCParametersObject *self, PyObject *args, PyObject *kwds) {
    // memory allocation
    if (self->base.params == NULL) {
        try {

            self->base.params = new moda::QEHCParameters();
        } catch (const std::bad_alloc& e) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for QEHCParameters.");
            return -1;
        }
    }
    
    // attribute parser
    int worseStyle = moda::SolverParameters::epsilon;
    int betterStyle = moda::SolverParameters::epsilon;
    int maxTime = 1000;
    int callbacks = 1; 
    unsigned long int iterLimit = 1000; // unsigned long int
    int sort = 1;
    int searchSubject = moda::QEHCParameters::MinimumContribution;
    int shuffle = 1;
    int offset = 2;

    if (!PyArg_ParseTuple(args, "|iiipliiip", 
                          &worseStyle, &betterStyle, &maxTime, &callbacks, 
                          &iterLimit, &sort, &searchSubject, &shuffle, &offset)) {
        return -1;
    }

    // attribute definition

    moda::QEHCParameters* qehc_params = (moda::QEHCParameters*)self->base.params;
    //derived attributes
    qehc_params->WorseReferencePointCalculationStyle = (moda::SolverParameters::ReferencePointCalculationStyle)worseStyle;
    qehc_params->BetterReferencePointCalculationStyle = (moda::SolverParameters::ReferencePointCalculationStyle)betterStyle;
    qehc_params->MaxEstimationTime = maxTime;
    qehc_params->callbacks = (bool)callbacks;

    //dedicated attributes
    qehc_params->iterationsLimit = iterLimit;
    qehc_params->sort = (bool)sort;
    qehc_params->SearchSubject = (moda::QEHCParameters::SearchSubjectOption)searchSubject;
    qehc_params->shuffle = (bool)shuffle;
    qehc_params->offset = offset;
    
    return 0;
}

// --- Get/Set for SearchSubject---
PyObject *QEHCParameters_get_SearchSubject(QEHCParametersObject *self, void *closure) {
    moda::QEHCParameters* qehc_params = (moda::QEHCParameters*)self->base.params;
    return PyLong_FromLong((long)qehc_params->SearchSubject);
}

int QEHCParameters_set_SearchSubject(QEHCParametersObject *self, PyObject *value, void *closure) {
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError, "Cannot delete the SearchSubject attribute");
        return -1;
    }
    long style_val = PyLong_AsLong(value);
    
    if (style_val < moda::QEHCParameters::MinimumContribution || style_val > moda::QEHCParameters::Both) {
        PyErr_SetString(PyExc_ValueError, "Invalid value for SearchSubjectOption.");
        return -1;
    }
    ((moda::QEHCParameters*)self->base.params)->SearchSubject = (moda::QEHCParameters::SearchSubjectOption)style_val;
    return 0;
}
PyObject* QEHCParameters_get_iterationsLimit(QEHCParametersObject *self, void *closure) {
    if (!self->base.params) { PyErr_SetString(PyExc_RuntimeError, "C++ object not init"); return NULL; }
    moda::QEHCParameters* cast = static_cast<moda::QEHCParameters*>(self->base.params);
    return PyLong_FromUnsignedLongLong(cast->iterationsLimit);
}

static int QEHCParameters_set_iterationsLimit(QEHCParametersObject *self, PyObject *value, void *closure) {
    if (!self->base.params) {
        PyErr_SetString(PyExc_RuntimeError, "Internal C++ parameters not initialized");
        return -1;
    }

    // Convert Python object to C++ type
    unsigned long val = PyLong_AsUnsignedLong(value);
    if (PyErr_Occurred()) return -1;

    // CAST: Access the member via the derived class pointer
    static_cast<moda::QEHCParameters*>(self->base.params)->iterationsLimit = val;
    
    return 0;
}
PyGetSetDef QEHCParameters_getsetters[] = {
    {"iterationsLimit", (getter)QEHCParameters_get_iterationsLimit, (setter)QEHCParameters_set_iterationsLimit, "Limit docs", NULL},
  
    {"SearchSubject", 
     (getter)QEHCParameters_get_SearchSubject, (setter)QEHCParameters_set_SearchSubject,
     "Type of the problem for the QEHCSolver contribution.", NULL},
    {NULL}  /* Sentinel */
};



PyTypeObject QEHCParametersType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "moda.QEHCParameters",                    /* tp_name */
    sizeof(QEHCParametersObject),             /* tp_basicsize */
    0,                                          /* tp_itemsize */
    0,                                          /* tp_dealloc */
    0,                                          /* tp_print (Deprecated) */
    0,                                          /* tp_getattr (Deprecated) */
    0,                                          /* tp_setattr (Deprecated) */
    0,                                          /* tp_compare (Deprecated) */
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number: Used for arithmetic operators (+, -, *, etc.) */                              
    0,                                          /* tp_as_sequence: Used for sequence protocols (tuple, list) */ 
    0,                                          /* tp_as_mapping: Used for indexing (p[i]) */ 
    0,                                          /* tp_hash */ 
    0,                                          /* tp_call */ 
    0,                                          /* tp_str: Used for str(p) */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags: KLUCZOWE dla dziedziczenia */
    "Parameters for the QEHC Solver.",          /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare: For comparisons (<, >, ==) */
    0,                                          /* tp_weaklistoffset */ 
    0,                                          /* tp_iter */ 
    0,                                          /* tp_iternext */ 
    NULL,                                       /* tp_methods (dla GetWorse/BetterReferencePoint) */
    0,                                          /* tp_members */
    QEHCParameters_getsetters,                  /* tp_getset */
    &SolverParametersType,                      /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    (initproc)QEHCParameters_init,            /* tp_init */
    0,                                          /* tp_alloc */
    (newfunc)PyType_GenericNew,                 /* tp_new */
};

#pragma endregion
#pragma region IQHV

int IQHVParameters_init(IQHVParameters *self, PyObject *args, PyObject *kwds) {
   
    // Proper memory allocation
    if (self->base.params == NULL) {
        try {
            // Allocate size of the derived object to the base object
            self->base.params = new moda::IQHVParameters();
        } catch (const std::bad_alloc& e) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for QEHCParameters.");
            return -1;
        }
    }
    
    // Attribute parsing
    //default values
    int worseStyle = moda::SolverParameters::epsilon;
    int betterStyle = moda::SolverParameters::epsilon;
    int maxTime = 1000;
    int callbacks = 1; 

    //parsing
    if (!PyArg_ParseTuple(args, "|iiipliiip", 
                          &worseStyle, &betterStyle, &maxTime, &callbacks)) {
        return -1;
    }

    // attributes definition
    // Derived attributes
    moda::IQHVParameters* iqhv_params = (moda::IQHVParameters*)self->base.params;
    
    iqhv_params->WorseReferencePointCalculationStyle = (moda::SolverParameters::ReferencePointCalculationStyle)worseStyle;
    iqhv_params->BetterReferencePointCalculationStyle = (moda::SolverParameters::ReferencePointCalculationStyle)betterStyle;
    iqhv_params->MaxEstimationTime = maxTime;
    iqhv_params->callbacks = (bool)callbacks;

    // IQHVParameters attributes

    // currently no dedicated params
    

    return 0;
}

PyGetSetDef IQHVParameters_getsetters[] = {
    //currently none
    {NULL}  /* Sentinel */
};


PyTypeObject IQHVParametersType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "moda.IQHVParameters",                    /* tp_name */
    sizeof(IQHVParametersObject),             /* tp_basicsize */
    0,                                          /* tp_itemsize */
    0,                                          /* tp_dealloc */
    0,                                          /* tp_print (Deprecated) */
    0,                                          /* tp_getattr (Deprecated) */
    0,                                          /* tp_setattr (Deprecated) */
    0,                                          /* tp_compare (Deprecated) */
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number: Used for arithmetic operators (+, -, *, etc.) */                              
    0,                                          /* tp_as_sequence: Used for sequence protocols (tuple, list) */ 
    0,                                          /* tp_as_mapping: Used for indexing (p[i]) */ 
    0,                                          /* tp_hash */ 
    0,                                          /* tp_call */ 
    0,                                          /* tp_str: Used for str(p) */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags: KLUCZOWE dla dziedziczenia */
    "Parameters for the IQHV Solver.",          /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare: For comparisons (<, >, ==) */
    0,                                          /* tp_weaklistoffset */ 
    0,                                          /* tp_iter */ 
    0,                                          /* tp_iternext */ 
    NULL,                                       /* tp_methods (dla GetWorse/BetterReferencePoint) */
    0,                                          /* tp_members */
    IQHVParameters_getsetters,                  /* tp_getset */
    &SolverParametersType,                      /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    (initproc)IQHVParameters_init,            /* tp_init */
    0,                                          /* tp_alloc */
    (newfunc)PyType_GenericNew,                 /* tp_new */
};
#pragma endregion

int HSSParameters_init(HSSParametersObject *self, PyObject *args, PyObject *kwds) {
    
    // Memory allocation
    if (self->base.params == NULL) {
        try {
            
            self->base.params = new moda::HSSParameters();
        } catch (const std::bad_alloc& e) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for QEHCParameters.");
            return -1;
        }
    }
    
    // attribute parser (with default values)
    int worseStyle = moda::SolverParameters::epsilon;
    int betterStyle = moda::SolverParameters::epsilon;
    int maxTime = 1000;
    int callbacks = 1; 
    int stoppingCriteria = 1;
    int selectionStrategy = 1;
    int stopSize = 100;
    int stopTime = 1000;

    if (!PyArg_ParseTuple(args, "|iiipiiii", 
                          &worseStyle, &betterStyle, &maxTime, &callbacks, 
                          &stoppingCriteria, &selectionStrategy, &stopSize, &stopTime)) {
        return -1;
    }

    // attribute definition
    moda::HSSParameters* hss_params = (moda::HSSParameters*)self->base.params;
    
    hss_params->WorseReferencePointCalculationStyle = (moda::SolverParameters::ReferencePointCalculationStyle)worseStyle;
    hss_params->BetterReferencePointCalculationStyle = (moda::SolverParameters::ReferencePointCalculationStyle)betterStyle;
    hss_params->MaxEstimationTime = maxTime;
    hss_params->callbacks = (bool)callbacks;

    // Pola klasy QEHCParameters
    hss_params->StoppingTime = stopTime;
    hss_params->StoppingSubsetSize = stopSize;
    hss_params->StoppingCriteria = (moda::IQHVParameters::StoppingCriteriaType)stoppingCriteria;
    hss_params->Strategy = (moda::IQHVParameters::SubsetSelectionStrategy)selectionStrategy;


    
    // (Pola 'maxlevel' nie ma w konstruktorze C++, ale możemy mu dać domyślną wartość 10)
    // qehc_params->maxlevel = 10;

    return 0;
}

// --- Get/Set for  ---
PyObject *HSSParameters_get_StoppingSubsetSize(HSSParametersObject *self, void *closure) {
    moda::HSSParameters* qehc_params = (moda::HSSParameters*)self->base.params;
    return PyLong_FromLong((long)qehc_params->StoppingSubsetSize);
}

int HSSParameters_set_StoppingSubsetSize(HSSParametersObject *self, PyObject *value, void *closure) {
    if (value == NULL) {
        PyErr_SetString(PyExc_AttributeError, "Cannot delete the StoppingSubsetSize attribute");
        return -1;
    }
    long val = PyLong_AsLong(value);
    
    if (val < 1) {
        PyErr_SetString(PyExc_ValueError, "Invalid value for StoppingSubsetSize.");
        return -1;
    }
    ((moda::QEHCParameters*)self->base.params)->StoppingSubsetSize =  style_val;
    return 0;
}
PyObject* HSSParameters_get_StoppingTime(HSSParametersObject *self, void *closure) {
    if (!self->base.params) { PyErr_SetString(PyExc_RuntimeError, "C++ object not init"); return NULL; }
    moda::HSSParameters* cast = static_cast<moda::HSSParameters*>(self->base.params);
    return PyLong_FromLong(cast->StoppingTime);
}

static int HSSParameters_set_StoppingTime(HSSParametersObject *self, PyObject *value, void *closure) {
    if (!self->base.params) {
        PyErr_SetString(PyExc_RuntimeError, "Internal C++ parameters not initialized");
        return -1;
    }

    // Convert Python object to C++ type
    long val = PyLong_AsLong(value);
    if (PyErr_Occurred()) return -1;

    // CAST: Access the member via the derived class pointer
    static_cast<moda::HSSParameters*>(self->base.params)->StoppingTime = val;
    
    return 0;
}
PyObject* HSSParameters_get_Strategy(HSSParametersObject *self, void *closure) {
    if (!self->base.params) { PyErr_SetString(PyExc_RuntimeError, "C++ object not init"); return NULL; }
    moda::HSSParameters* cast = static_cast<moda::HSSParameters*>(self->base.params);
    return PyLong_FromLong(cast->Strategy);
}
static int HSSParameters_set_Strategy(HSSParametersObject *self, PyObject *value, void *closure) {
    if (!self->base.params) {
        PyErr_SetString(PyExc_RuntimeError, "Internal C++ parameters not initialized");
        return -1;
    }

    // Convert Python object to C++ type
    long style_val = PyLong_AsLong(value);
    if (PyErr_Occurred()) return -1;

    // CAST: Access the member via the derived class pointer
    static_cast<moda::HSSParameters*>(self->base.params)->Strategy = (moda::HSSParameters::SubsetSelectionStrategy)style_val;;
    
    return 0;
}
PyObject* HSSParameters_get_Criteria(HSSParametersObject *self, void *closure) {
    if (!self->base.params) { PyErr_SetString(PyExc_RuntimeError, "C++ object not init"); return NULL; }
    moda::HSSParameters* cast = static_cast<moda::HSSParameters*>(self->base.params);
    return PyLong_FromLong(cast->StoppingCriteria);
}
static int HSSParameters_set_Criteria(HSSParametersObject *self, PyObject *value, void *closure) {
    if (!self->base.params) {
        PyErr_SetString(PyExc_RuntimeError, "Internal C++ parameters not initialized");
        return -1;
    }

    // Convert Python object to C++ type
    long style_val = PyLong_AsLong(value);
    if (PyErr_Occurred()) return -1;

    // CAST: Access the member via the derived class pointer
    static_cast<moda::HSSParameters*>(self->base.params)->StoppingCriteria = (moda::HSSParameters::StoppingCriteriaType)style_val;;
    
    return 0;
}
PyGetSetDef HSSParameters_getsetters[] = {
    {"Strategy", (getter)HSSParameters_get_Strategy, (setter)HSSParameters_set_Strategy, "Limit docs", NULL},
    {"Criteria", (getter)HSSParameters_get_Criteria, (setter)HSSParameters_set_Criteria, "Limit docs", NULL},
    {"StoppingSize", (getter)HSSParameters_get_StoppingSubsetSize, (setter)HSSParameters_set_StoppingTime,"Type of the problem for the QEHCSolver contribution.", NULL},
    {"StoppingTime", (getter)HSSParameters_get_StoppingTime, (setter)HSSParameters_set_StoppingTime,"Type of the problem for the QEHCSolver contribution.", NULL},
    {NULL}  /* Sentinel */
};



PyTypeObject HSSParametersType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "moda.HSSParameters",                    /* tp_name */
    sizeof(HSSParametersObject),             /* tp_basicsize */
    0,                                          /* tp_itemsize */
    0,                                          /* tp_dealloc */
    0,                                          /* tp_print (Deprecated) */
    0,                                          /* tp_getattr (Deprecated) */
    0,                                          /* tp_setattr (Deprecated) */
    0,                                          /* tp_compare (Deprecated) */
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number: Used for arithmetic operators (+, -, *, etc.) */                              
    0,                                          /* tp_as_sequence: Used for sequence protocols (tuple, list) */ 
    0,                                          /* tp_as_mapping: Used for indexing (p[i]) */ 
    0,                                          /* tp_hash */ 
    0,                                          /* tp_call */ 
    0,                                          /* tp_str: Used for str(p) */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags: KLUCZOWE dla dziedziczenia */
    "Parameters for the HSS Solver.",          /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare: For comparisons (<, >, ==) */
    0,                                          /* tp_weaklistoffset */ 
    0,                                          /* tp_iter */ 
    0,                                          /* tp_iternext */ 
    NULL,                                       /* tp_methods (dla GetWorse/BetterReferencePoint) */
    0,                                          /* tp_members */
    HSSParameters_getsetters,                  /* tp_getset */
    &SolverParametersType,                      /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    (initproc)HSSParameters_init,            /* tp_init */
    0,                                          /* tp_alloc */
    (newfunc)PyType_GenericNew,                 /* tp_new */
};
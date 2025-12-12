#pragma once
#include "../SolverParameters.h"
#include <structmember.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include "py_point.cpp"
typedef struct {
    PyObject_HEAD
    moda::SolverParameters *params;
} SolverParametersObject;

typedef struct {
    SolverParametersObject base; // Musi być PIERWSZE dla dziedziczenia w Pythonie!
    moda::QEHCParameters *innerParams;
} QEHCParametersObject;

// Definicja typu (który jest faktycznie tylko przestrzenią nazw dla stałych)
static PyTypeObject ReferencePointCalculationStyleType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "moda.ReferencePointCalculationStyle",
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 17 zer dla tp_basicsize do tp_iternext
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,                                   // tp_flags
    "Styl kalkulacji punktu referencyjnego (lepszego/gorszego)."
};



// Definicja typu dla enuma dla qehc
static PyTypeObject SearchSubjectOptionType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "moda.QEHCParameters.SearchSubjectOption",
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, 
    "Typ problemu dla wkładu QEHCSolver (minimalny, maksymalny lub oba).",
};

// Funkcja pomocnicza do inicjalizacji enuma 
static int init_ReferencePointCalculationStyle(PyObject *m) {
    PyObject *enum_type_obj = (PyObject *)&ReferencePointCalculationStyleType;

    // 1. Zapewniamy, że typ jest w module (to jest OK)
    Py_INCREF(enum_type_obj);
    if (PyModule_AddObject(m, "ReferencePointCalculationStyle", enum_type_obj) < 0) {
        Py_DECREF(enum_type_obj);
        return -1;
    }
    
    // 2. Pobranie i modyfikacja słownika typu (tp_dict)
    PyObject *dict = ((PyTypeObject *)enum_type_obj)->tp_dict;
    if (!dict) {
        PyErr_SetString(PyExc_RuntimeError, "Type dictionary is NULL.");
        return -1;
    }

    // Makro pomocnicze do dodawania stałych bezpośrednio do słownika typu
    #define ADD_ENUM_CONST_TO_DICT(name, val) \
        do { \
            PyObject *v = PyLong_FromLong(val); \
            if (!v) return -1; \
            int result = PyDict_SetItemString(dict, name, v); \
            Py_DECREF(v); \
            if (result < 0) return -1; \
        } while (0)

    // Dodanie stałych
    ADD_ENUM_CONST_TO_DICT("epsilon", moda::SolverParameters::epsilon);
    ADD_ENUM_CONST_TO_DICT("tenpercent", moda::SolverParameters::tenpercent);
    ADD_ENUM_CONST_TO_DICT("zeroone", moda::SolverParameters::zeroone);
    ADD_ENUM_CONST_TO_DICT("userdefined", moda::SolverParameters::userdefined);
    ADD_ENUM_CONST_TO_DICT("exact", moda::SolverParameters::exact);

    #undef ADD_ENUM_CONST_TO_DICT
    
    return 0;
}
// Funkcja pomocnicza do inicjalizacji enuma (wywołana w PyInit_moda)
static int init_SearchSubjectOption(PyObject *m) {
    // Obiekt typu, do którego dodajemy stałe
    PyObject *enum_type_obj = (PyObject *)&SearchSubjectOptionType;
    
    // 1. Upewnienie się, że typ jest gotowy (już wywołane w PyInit_moda, ale bezpieczne do powtórzenia)
    if (PyType_Ready(&SearchSubjectOptionType) < 0) return -1;
    
    // 2. Dodanie typu do modułu m 
    Py_INCREF(enum_type_obj);
    if (PyModule_AddObject(m, "SearchSubjectOption", enum_type_obj) < 0) {
        Py_DECREF(enum_type_obj);
        return -1;
    }

    // --- NOWY FRAGMENT: BEZPOŚREDNIA MODYFIKACJA SŁOWNIKA TYPU ---
    
    // 3. Pobranie słownika typu (tp_dict)
    PyObject *dict = ((PyTypeObject *)enum_type_obj)->tp_dict;
    if (!dict) {
        PyErr_SetString(PyExc_RuntimeError, "Type dictionary is NULL for SearchSubjectOptionType.");
        return -1;
    }

    // Makro pomocnicze do dodawania stałych bezpośrednio do słownika typu
    #define ADD_ENUM_CONST_TO_DICT(name, val) \
        do { \
            /* Tworzenie obiektu PyLong z wartości int */ \
            PyObject *v = PyLong_FromLong(val); \
            if (!v) return -1; \
            /* Dodawanie do słownika typu za pomocą PyDict_SetItemString */ \
            int result = PyDict_SetItemString(dict, name, v); \
            Py_DECREF(v); \
            if (result < 0) return -1; \
        } while (0)

    // 4. Dodanie stałych
    ADD_ENUM_CONST_TO_DICT("MinimumContribution", moda::QEHCParameters::MinimumContribution);
    ADD_ENUM_CONST_TO_DICT("MaximumContribution", moda::QEHCParameters::MaximumContribution);
    ADD_ENUM_CONST_TO_DICT("Both", moda::QEHCParameters::Both);
    
    #undef ADD_ENUM_CONST_TO_DICT
    
    return 0;
}

static void SolverParameters_dealloc(SolverParametersObject *self) {
    if (self->params) {
        delete self->params;
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int SolverParameters_init(SolverParametersObject *self, PyObject *args, PyObject *kwds) {
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



static PyObject *SolverParameters_get_WorseStyle(SolverParametersObject *self, void *closure) {
    return PyLong_FromLong((long)self->params->WorseReferencePointCalculationStyle);
}

static int SolverParameters_set_WorseStyle(SolverParametersObject *self, PyObject *value, void *closure) {
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

static PyObject *SolverParameters_get_BetterStyle(SolverParametersObject *self, void *closure) {
    return PyLong_FromLong((long)self->params->BetterReferencePointCalculationStyle);
}

static int SolverParameters_set_BetterStyle(SolverParametersObject *self, PyObject *value, void *closure) {
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


static PyObject *SolverParameters_get_betterRefPoint(SolverParametersObject *self, void *closure) {
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

static PyObject *SolverParameters_get_worseRefPoint(SolverParametersObject *self, void *closure) {
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

static PyGetSetDef SolverParameters_getsetters[] = {
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

static PyMemberDef SolverParameters_members[] = {
    {"callbacks", 
     T_BOOL, 
     offsetof(SolverParametersObject, params) + offsetof(moda::SolverParameters, callbacks), 
     0,
     "should the solver use iteration callbacks"},
     
    {"MaxEstimationTime", 
     T_INT, 
     offsetof(SolverParametersObject, params) + offsetof(moda::SolverParameters, MaxEstimationTime), 
     0,
     "maximum time of estimation in ms"},
     
    {"seed", 
     T_UINT, // unsigned int
     offsetof(SolverParametersObject, params) + offsetof(moda::SolverParameters, seed), 
     0,
     "Custom random seed"},
     
    {NULL}
};

static PyTypeObject SolverParametersType = {
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
    SolverParameters_members,                   /* tp_members */
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



static int QEHCParameters_init(QEHCParametersObject *self, PyObject *args, PyObject *kwds) {
    // 1. Wywołanie konstruktora klasy bazowej (SolverParameters_init)
    // Zapewniamy, że pole 'base.params' zostanie zainicjalizowane na moda::SolverParameters*.
    // Jeśli nie wywołamy init klasy bazowej, musimy sami zaalokować pamięć i ją skonstruować.
    
    // Alokacja pamięci dla QEHCParameters* (zamiast SolverParameters*)
    if (self->base.params == NULL) {
        try {
            // Alokujemy pełną klasę pochodną, ale przypisujemy do wskaźnika klasy bazowej.
            self->base.params = new moda::QEHCParameters();
        } catch (const std::bad_alloc& e) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for QEHCParameters.");
            return -1;
        }
    }
    
    // Parser argumentów: wszystkie opcjonalne, tak jak w konstruktorze C++
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

    // Ustawienie pól C++ na podstawie argumentów Pythona:
    // Pola odziedziczone są już w self->base.params, który wskazuje na QEHCParameters*
    moda::QEHCParameters* qehc_params = (moda::QEHCParameters*)self->base.params;
    
    qehc_params->WorseReferencePointCalculationStyle = (moda::SolverParameters::ReferencePointCalculationStyle)worseStyle;
    qehc_params->BetterReferencePointCalculationStyle = (moda::SolverParameters::ReferencePointCalculationStyle)betterStyle;
    qehc_params->MaxEstimationTime = maxTime;
    qehc_params->callbacks = (bool)callbacks;

    // Pola klasy QEHCParameters
    qehc_params->iterationsLimit = iterLimit;
    qehc_params->sort = (bool)sort;
    qehc_params->SearchSubject = (moda::QEHCParameters::SearchSubjectOption)searchSubject;
    qehc_params->shuffle = (bool)shuffle;
    qehc_params->offset = offset;
    
    // (Pola 'maxlevel' nie ma w konstruktorze C++, ale możemy mu dać domyślną wartość 10)
    // qehc_params->maxlevel = 10;

    return 0;
}

// --- Get/Set dla SearchSubject (Enum) ---
static PyObject *QEHCParameters_get_SearchSubject(QEHCParametersObject *self, void *closure) {
    moda::QEHCParameters* qehc_params = (moda::QEHCParameters*)self->base.params;
    return PyLong_FromLong((long)qehc_params->SearchSubject);
}

static int QEHCParameters_set_SearchSubject(QEHCParametersObject *self, PyObject *value, void *closure) {
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

static PyGetSetDef QEHCParameters_getsetters[] = {
    {"SearchSubject", 
     (getter)QEHCParameters_get_SearchSubject, (setter)QEHCParameters_set_SearchSubject,
     "Type of the problem for the QEHCSolver contribution.", NULL},
    {NULL}  /* Sentinel */
};


// --- Mapowanie Pól (PyMemberDef) ---
// Używamy T_ULONGLONG dla unsigned long int (lub T_ULONG), T_BOOL, T_INT
static PyMemberDef QEHCParameters_members[] = {
    {"iterationsLimit", 
     T_ULONGLONG, // lub T_ULONG, zależy od platformy
     offsetof(QEHCParametersObject, base.params) + offsetof(moda::QEHCParameters, iterationsLimit), 
     0,
     "iterations limit for QEHCSolver"},
     
    {"shuffle", 
     T_BOOL, 
     offsetof(QEHCParametersObject, base.params) + offsetof(moda::QEHCParameters, shuffle), 
     0,
     "If sorting is not allowed, should the set be shuffled."},
     
    {"offset", 
     T_INT, 
     offsetof(QEHCParametersObject, base.params) + offsetof(moda::QEHCParameters, offset), 
     0,
     "If set is being rotated, indicates rotation offset."},
     
    {"sort", 
     T_BOOL, 
     offsetof(QEHCParametersObject, base.params) + offsetof(moda::QEHCParameters, sort), 
     0,
     "Is sorting allowed in QEHCSolver."},
     
    {"maxlevel", 
     T_INT, 
     offsetof(QEHCParametersObject, base.params) + offsetof(moda::QEHCParameters, maxlevel), 
     0,
     "Maximum level parameter."},
     
    {NULL}  // Sentinel
};

static PyTypeObject QEHCParametersType = {
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
    QEHCParameters_members,                     /* tp_members */
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
#pragma once

#include <Python.h>
#include "../Solver.h" // Zawiera definicję moda::Solver
#include "moda_types.h"



extern PyTypeObject SolverType;

// Dealokator
void Solver_dealloc(SolverObject *self) {
    // Ponieważ Solver jest klasą abstrakcyjną, usuwamy tylko, jeśli wskaźnik
    // nie jest NULL i NIE JESTEŚMY w konstruktorze klasy pochodnej
    if (self->solver != NULL) {
        // !!! WAŻNE: Tutaj nie usuwamy, ponieważ klasy pochodne (QEHCSolver)
        // będą tworzyły obiekt. Zostawiamy dealokację klasom pochodnym.
        // Jeśli będziemy używać tego typu do dziedziczenia, 
        // musimy ostrożnie zarządzać pamięcią.
        
        // Na razie zostawiamy, zakładając, że konstruktor nie alokuje pamięci
        // (Base constructor Solver() jest pusty, więc nie alokujemy tutaj)
    }
    
    // Zwolnienie referencji do Callbacks
    Py_XDECREF(self->start_callback);
    Py_XDECREF(self->iteration_callback);
    Py_XDECREF(self->end_callback);

    Py_TYPE(self)->tp_free((PyObject *) self);
}

// Konstruktor/Alokator (Blokowanie bezpośredniej instancjacji)
PyObject *Solver_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    // Blokujemy instancjację klasy bazowej Solver, jeśli nie jest to klasa pochodna
    if (type == &SolverType) {
        PyErr_SetString(PyExc_TypeError, "Cannot instantiate abstract base class 'Solver'.");
        return NULL;
    }

    // Pozwalamy na alokację, jeśli jest to klasa pochodna
    SolverObject *self = (SolverObject *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->solver = NULL; // Wskaźnik C++ jest inicjowany w klasach pochodnych
        self->start_callback = Py_None; Py_INCREF(Py_None);
        self->iteration_callback = Py_None; Py_INCREF(Py_None);
        self->end_callback = Py_None; Py_INCREF(Py_None);
    }
    return (PyObject *)self;
}

// --- Gettery i Settery dla pól ---

PyObject *Solver_get_currentlySolvedProblem(SolverObject *self, void *closure) {
    if (self->solver == NULL || self->solver->currentlySolvedProblem == NULL) {
        Py_RETURN_NONE;
    }
    // UWAGA: Trzeba by stworzyć funkcję tworzącą DatasetObject z istniejącego wskaźnika
    // Wymaga funkcji Dataset_create_from_pointer(moda::DataSet* ptr)
    // Na potrzeby tego przykładu zwrócimy None, jeśli nie mamy tej funkcji
    Py_RETURN_NONE; 
}

PyObject *Solver_get_currentSettings(SolverObject *self, void *closure) {
    if (self->solver == NULL || self->solver->currentSettings == NULL) {
        Py_RETURN_NONE;
    }
    // Podobnie, wymaga funkcji DatasetParameters_create_from_pointer
    Py_RETURN_NONE; 
}

PyGetSetDef Solver_getsetters[] = {
    {"currentlySolvedProblem", 
     (getter)Solver_get_currentlySolvedProblem, NULL,
     "The current problem being solved (DataSet*).", NULL},
    {"currentSettings", 
     (getter)Solver_get_currentSettings, NULL,
     "Current solver settings (DataSetParameters*).", NULL},
    
    // --- Callbacks ---
    {"StartCallback", 
     (getter)PyObject_GenericGetAttr, (setter)PyObject_GenericSetAttr,
     "Callback function executed at the start of solving.", NULL},
    {"IterationCallback", 
     (getter)PyObject_GenericGetAttr, (setter)PyObject_GenericSetAttr,
     "Callback function executed after each iteration.", NULL},
    {"EndCallback", 
     (getter)PyObject_GenericGetAttr, (setter)PyObject_GenericSetAttr,
     "Callback function executed at the end of solving.", NULL},
    
    {NULL}  /* Sentinel */
};

PyTypeObject SolverType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "moda.Solver",                 /* tp_name */
    sizeof(SolverObject),          /* tp_basicsize */
    0,                             /* tp_itemsize */
    (destructor)Solver_dealloc,    /* tp_dealloc */
    0, /* tp_print (Deprecated) */
    0, /* tp_getattr (Deprecated) */
    0, /* tp_setattr (Deprecated) */
    0, /* tp_compare (Deprecated) */
    0, /* tp_repr (Optional: string representation) */
    0, /* tp_as_number: Used for arithmetic operators (+, -, *, etc.) */
    0, /* tp_as_sequence: Used for sequence protocols (tuple, list) */
    0, /* tp_as_mapping: Used for indexing (p[i]) */
    0, /* tp_hash */
    0, /* tp_call */
    0, /* tp_str: Used for str(p) */
    0, /* tp_getattro */
    0, /* tp_setattro */
    0, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags: KLUCZOWE dla dziedziczenia */
    "Base abstract class for optimization solvers.", /* tp_doc */
    0, /* tp_traverse */
    0, /* tp_clear */
    0, /* tp_richcompare: For comparisons (<, >, ==) */
    0, /* tp_weaklistoffset */
    0, /* tp_iter */
    0, /* tp_iternext */
    NULL,                          /* tp_methods (brak, bo są wirtualne/abstrakcyjne) */
    NULL,                          /* tp_members */
    Solver_getsetters,             /* tp_getset */
    0,                             /* tp_base */                           
    0,                             /* tp_dict */
    0,                             /* tp_descr_get */
    0,                             /* tp_descr_set */
    0,                             /* tp_dictoffset */
    0,                             /* tp_init (brak, bo blokujemy instancjację) */
    0,                             /* tp_alloc */
    (newfunc)Solver_new,           /* tp_new */
};

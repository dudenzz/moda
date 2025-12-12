// qehc_solver_wrap.h
#include "py_solver.cpp" // Zawiera definicję SolverObject i SolverType
#include "../QEHCSolver.cpp"        // Zawiera definicję moda::QEHCSolver
#include "../Result.h"
#include "py_dataset.cpp"
#include "py_solver_parameters.cpp"
// Struktura dla QEHC Solver (dziedziczy z SolverObject)
typedef struct {
    // Odziedziczone pola z SolverObject (PyObject_HEAD i moda::Solver *solver)
    SolverObject super; 
    // Brak dodatkowych pól C++ do opakowania w tej klasie pochodnej
} QEHCSolverObject;

extern PyTypeObject QEHCSolverType;

// Konstruktor/Alokator dla QEHCSolver
static PyObject *QEHCSolver_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    QEHCSolverObject *self;
    
    // Używamy tp_alloc typu bazowego SolverObject, ponieważ QEHCSolverObject to rozszerzenie
    self = (QEHCSolverObject *)SolverType.tp_new(type, args, kwds);

    if (self != NULL) {
        // Alokacja obiektu C++ QEHCSolver
        try {
            // Używamy pola 'solver' z odziedziczonej struktury SolverObject
            self->super.solver = new moda::QEHCSolver(); 
        } catch (const std::bad_alloc &e) {
            Py_DECREF(self);
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for moda::QEHCSolver.");
            return NULL;
        } catch (...) {
            Py_DECREF(self);
            PyErr_SetString(PyExc_RuntimeError, "Unknown error during QEHCSolver allocation.");
            return NULL;
        }
    }
    return (PyObject *)self;
}

// Inicjalizator
static int QEHCSolver_init(QEHCSolverObject *self, PyObject *args, PyObject *kwds) {
    // Jeśli potrzebna jest inicjalizacja z argumentami (np. ustawienia domyślne),
    // tutaj jest miejsce na parsowanie args/kwds
    return 0;
}

// Dealokator
static void QEHCSolver_dealloc(QEHCSolverObject *self) {
    // Dealokacja obiektu C++
    if (self->super.solver != NULL) {
        // Bezpieczne rzutowanie i usunięcie obiektu
        delete self->super.solver;
        self->super.solver = NULL;
    }
    
    // Wywołanie dealokatora typu bazowego (aby zwolnić pola callbacks itd.)
    SolverType.tp_dealloc((PyObject *)self);
}

// --- Metoda Solve ---
static PyObject *QEHCSolver_Solve(QEHCSolverObject *self, PyObject *args) {
    PyObject *py_dataset_obj;
    PyObject *py_params_obj;
    
    // Parsowanie argumentów: Oczekujemy DataSet i QEHCParameters
    if (!PyArg_ParseTuple(args, "OO", &py_dataset_obj, &py_params_obj)) {
        return NULL;
    }

    // Walidacja typów (tutaj zakładamy, że mamy ich PyTypeObject)
    /*
    if (!PyObject_IsInstance(py_dataset_obj, (PyObject *)&DatasetType) || 
        !PyObject_IsInstance(py_params_obj, (PyObject *)&QEHCParametersType)) {
        PyErr_SetString(PyExc_TypeError, "Solve requires a DataSet and QEHCParameters object.");
        return NULL;
    }
    */

    // Rzutowanie na obiekty wrapperów
    DataSetObject *dataset_wrapper = (DataSetObject *)py_dataset_obj;
    QEHCParametersObject *params_wrapper = (QEHCParametersObject *)py_params_obj;

    // Walidacja wewnętrznych wskaźników C++
    if (!self->super.solver || !dataset_wrapper->data_set || !params_wrapper->base.params) {
        PyErr_SetString(PyExc_RuntimeError, "Internal C++ object or parameters were not initialized.");
        return NULL;
    }

    moda::QEHCResult* result_ptr = NULL;
    moda::QEHCParameters* params =  new moda::QEHCParameters(moda::QEHCParameters::ReferencePointCalculationStyle::exact, moda::QEHCParameters::ReferencePointCalculationStyle::exact);
    try {
        // Wywołanie metody C++
        // Uwaga: Funkcja Solve przyjmuje QEHCParameters przez wartość, więc musimy przekazać kopię
        result_ptr = ((moda::QEHCSolver*)self->super.solver)->Solve(
            dataset_wrapper->data_set, 
            *params // Dereferencja, aby przekazać przez wartość
        );
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }

    // Tworzenie obiektu Pythona z wyniku (wymaga funkcji fabrycznej QEHCResult_create_from_pointer)
    // return QEHCResult_create_from_pointer(result_ptr);
    
    // Tymczasowo zwracamy None
    // if (result_ptr) {
    //     // UWAGA: Potrzebna dealokacja result_ptr jeśli to Solve alokuje pamięć!
    //     // delete result_ptr; 
    // }
    Py_RETURN_NONE;
}

// Tabela metod
static PyMethodDef QEHCSolver_methods[] = {
    {"Solve", (PyCFunction)QEHCSolver_Solve, METH_VARARGS,
     "Solves the optimization problem using QEHC algorithm."},
    {NULL}  /* Sentinel */
};

static PyTypeObject QEHCSolverType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "moda.QEHCSolver",             /* tp_name */
    sizeof(QEHCSolverObject),      /* tp_basicsize */
    0, 
    (destructor)QEHCSolver_dealloc, /* tp_dealloc (używa własnego dealokatora) */
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
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    "Solver implementation based on QEHCSolver.", /* tp_doc */
    0, /* tp_traverse */
    0, /* tp_clear */
    0, /* tp_richcompare: For comparisons (<, >, ==) */
    0, /* tp_weaklistoffset */
    0, /* tp_iter */
    0, /* tp_iternext */
    QEHCSolver_methods,            /* tp_methods */
    NULL,                          /* tp_members */
    NULL,                          /* tp_getset */
    &SolverType,                   /* tp_base: DZIEDZICZY Z SolverType */
    0, /* tp_dict */
    0, /* tp_descr_get */
    0, /* tp_descr_set */
    0, /* tp_dictoffset */
    (initproc)QEHCSolver_init,     /* tp_init */
    0, 
    (newfunc)QEHCSolver_new,       /* tp_new */
};


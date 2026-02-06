// qehc_solver_wrap.h
#include "../QEHCSolver.h"        // Zawiera definicję moda::QEHCSolver
#include "../Result.h"
#include "moda_types.h"





// Konstruktor/Alokator dla QEHCSolver
PyObject *QEHCSolver_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {

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

// initializer
int QEHCSolver_init(QEHCSolverObject *self, PyObject *args, PyObject *kwds) {
    self->super.solver = new moda::QEHCSolver();
    return 0;
}

// deallocator
void QEHCSolver_dealloc(QEHCSolverObject *self) {
    if (self->super.solver) {
        delete self->super.solver; // This triggers the chain of destructors
        self->super.solver = nullptr;
    }

    Py_TYPE(self)->tp_free((PyObject *)self);
}

// --- Solve method ---
PyObject *QEHCSolver_Solve(QEHCSolverObject *self, PyObject *args) {
    PyObject *py_dataset_obj;
    PyObject *py_params_obj;

    // Argument parsing
    if (!PyArg_ParseTuple(args, "OO", &py_dataset_obj, &py_params_obj)) {
        return NULL;
    }


    // wrapping the python object in the API object type
    DataSetObject *dataset_wrapper = (DataSetObject *)py_dataset_obj;
    if (!dataset_wrapper->data_set) {
        fprintf(stderr, "CRITICAL: DataSet C++ pointer is NULL!\n");
        fflush(stderr);
        return NULL;
    }

    
    QEHCParametersObject *params_wrapper = (QEHCParametersObject *)py_params_obj;   
    // SolverParametersObject *base_params_wrapper = (SolverParametersObject *)py_params_obj;  
    // pointers validation
    if (!self->super.solver || !dataset_wrapper->data_set || !params_wrapper->base.params) {
        PyErr_SetString(PyExc_RuntimeError, "Internal C++ object or parameters were not initialized.");
        return NULL;
    }

    moda::QEHCResult* result_ptr = NULL;
    moda::QEHCParameters* params =  new moda::QEHCParameters(moda::QEHCParameters::ReferencePointCalculationStyle::pymoo, moda::QEHCParameters::ReferencePointCalculationStyle::pymoo);
    params->SearchSubject = moda::QEHCParameters::SearchSubjectOption::MinimumContribution;
    try {

        params->iterationsLimit = static_cast<moda::QEHCParameters*>(params_wrapper->base.params)->iterationsLimit;
        params->shuffle = static_cast<moda::QEHCParameters*>(params_wrapper->base.params)->shuffle;
        params->offset= static_cast<moda::QEHCParameters*>(params_wrapper->base.params)->offset;
        params->SearchSubject= static_cast<moda::QEHCParameters*>(params_wrapper->base.params)->SearchSubject;
        params->sort= static_cast<moda::QEHCParameters*>(params_wrapper->base.params)->sort;
        params->maxlevel= static_cast<moda::QEHCParameters*>(params_wrapper->base.params)->maxlevel;
        // params->WorseReferencePointCalculationStyle = static_cast<moda::SolverParameters*>(base_params_wrapper->params)->WorseReferencePointCalculationStyle;
        // params->BetterReferencePointCalculationStyle = static_cast<moda::SolverParameters*>(base_params_wrapper->params)->BetterReferencePointCalculationStyle;        //static cast
        moda::QEHCSolver* qehc_ptr = static_cast<moda::QEHCSolver*>(self->super.solver);
        // calling the method
        result_ptr = qehc_ptr->Solve(
            dataset_wrapper->data_set, 
            *params
        );

        delete params;


    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }


    

    // Create a tuple of size 5
    PyObject* pyTuple = PyTuple_New(5);

    PyTuple_SetItem(pyTuple, 0, PyFloat_FromDouble(result_ptr->MinimumContribution));
    PyTuple_SetItem(pyTuple, 1, PyFloat_FromDouble(result_ptr->MaximumContribution));
    PyTuple_SetItem(pyTuple, 2, PyLong_FromLong(result_ptr->MinimumContributionIndex));
    PyTuple_SetItem(pyTuple, 3, PyLong_FromLong(result_ptr->MaximumContributionIndex));
    PyTuple_SetItem(pyTuple, 4, PyLong_FromLong(result_ptr->ElapsedTime));
    delete result_ptr;
    return pyTuple;
}

// methods
PyMethodDef QEHCSolver_methods[] = {
    {"Solve", (PyCFunction)QEHCSolver_Solve, METH_VARARGS,
     "Solves the optimization problem using QEHC algorithm."},
    {NULL}  /* Sentinel */
};

PyTypeObject QEHCSolverType = {
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


// iqhv_solver_wrap.h
#include "../IQHVSolver.h"        // Zawiera definicję moda::IQHV
#include "../Result.h"
#include "moda_types.h"





// initializer
PyObject *IQHVSolver_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {

    IQHVSolverObject *self;
    
    //memory allocation
    self = (IQHVSolverObject *)SolverType.tp_new(type, args, kwds);
    if (self != NULL) {
        //
        try {
            // the base solver is used to store the object of a derived class
            self->super.solver = new moda::IQHVSolver(); 
        } catch (const std::bad_alloc &e) {
            Py_DECREF(self);
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for moda::IQHVSolver.");
            return NULL;
        } catch (...) {
            Py_DECREF(self);
            PyErr_SetString(PyExc_RuntimeError, "Unknown error during IQHVSolver allocation.");
            return NULL;
        }
    }
    return (PyObject *)self;
}

// initializer
int IQHVSolver_init(IQHVSolverObject *self, PyObject *args, PyObject *kwds) {
    self->super.solver = new moda::IQHVSolver();
    return 0;
}

// deallocator
void IQHVSolver_dealloc(IQHVSolverObject *self) {
    if (self->super.solver) {
        delete self->super.solver; // This triggers the chain of destructors
        self->super.solver = nullptr;
    }

    Py_TYPE(self)->tp_free((PyObject *)self);
}

// --- Solve method ---
PyObject *IQHVSolver_Solve(IQHVSolverObject *self, PyObject *args) {
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

    
    IQHVParametersObject *params_wrapper = (IQHVParametersObject *)py_params_obj;   

    // pointers validation
    if (!self->super.solver || !dataset_wrapper->data_set || !params_wrapper->base.params) {
        PyErr_SetString(PyExc_RuntimeError, "Internal C++ object or parameters were not initialized.");
        return NULL;
    }

    moda::IQHVResult* result_ptr = NULL;
    moda::IQHVParameters* params =  new moda::IQHVParameters(moda::IQHVParameters::ReferencePointCalculationStyle::exact, moda::IQHVParameters::ReferencePointCalculationStyle::exact);
    try {


        //static cast
        moda::IQHVSolver* iqhv_ptr = static_cast<moda::IQHVSolver*>(self->super.solver);
        // calling the method
        result_ptr = iqhv_ptr->Solve(
            dataset_wrapper->data_set, 
            *params
        );

        delete params;


    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }

    // Create a tuple of size 2
    PyObject* pyTuple = PyTuple_New(2);
    PyTuple_SetItem(pyTuple, 0, PyFloat_FromDouble(result_ptr->HyperVolume));
    PyTuple_SetItem(pyTuple, 1, PyLong_FromLong(result_ptr->ElapsedTime));
    delete result_ptr;
    return pyTuple;
}

// methods
PyMethodDef IQHVSolver_methods[] = {
    {"Solve", (PyCFunction)IQHVSolver_Solve, METH_VARARGS,
     "Solves the optimization problem using IQHV algorithm."},
    {NULL}  /* Sentinel */
};

PyTypeObject IQHVSolverType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "moda.IQHVSolver",             /* tp_name */
    sizeof(IQHVSolverObject),      /* tp_basicsize */
    0, 
    (destructor)IQHVSolver_dealloc, /* tp_dealloc (używa własnego dealokatora) */
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
    "Solver implementation based on IQHVSolver.", /* tp_doc */
    0, /* tp_traverse */
    0, /* tp_clear */
    0, /* tp_richcompare: For comparisons (<, >, ==) */
    0, /* tp_weaklistoffset */
    0, /* tp_iter */
    0, /* tp_iternext */
    IQHVSolver_methods,            /* tp_methods */
    NULL,                          /* tp_members */
    NULL,                          /* tp_getset */
    &SolverType,                   /* tp_base: DZIEDZICZY Z SolverType */
    0, /* tp_dict */
    0, /* tp_descr_get */
    0, /* tp_descr_set */
    0, /* tp_dictoffset */
    (initproc)IQHVSolver_init,     /* tp_init */
    0, 
    (newfunc)IQHVSolver_new,       /* tp_new */
};


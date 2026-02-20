// hss_solver_wrap.h
#include "../HSSSolver.h"        // Zawiera definicję moda::HSSSolver
#include "../Result.h"
#include "moda_types.h"





// Konstruktor/Alokator dla HSSSolver
PyObject *HSSSolver_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {

    HSSSolverObject *self;
    
    // Używamy tp_alloc typu bazowego SolverObject, ponieważ HSSSolverObject to rozszerzenie
    self = (HSSSolverObject *)SolverType.tp_new(type, args, kwds);
    if (self != NULL) {
        // Alokacja obiektu C++ HSSSolver
        try {
            // Używamy pola 'solver' z odziedziczonej struktury SolverObject
            self->super.solver = new moda::HSSSolver(); 
        } catch (const std::bad_alloc &e) {
            Py_DECREF(self);
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for moda::HSSSolver.");
            return NULL;
        } catch (...) {
            Py_DECREF(self);
            PyErr_SetString(PyExc_RuntimeError, "Unknown error during HSSSolver allocation.");
            return NULL;
        }
    }
    return (PyObject *)self;
}

// initializer
int HSSSolver_init(HSSSolverObject *self, PyObject *args, PyObject *kwds) {
    self->super.solver = new moda::HSSSolver();
    return 0;
}

// deallocator
void HSSSolver_dealloc(HSSSolverObject *self) {
    if (self->super.solver) {
        delete self->super.solver; // This triggers the chain of destructors
        self->super.solver = nullptr;
    }

    Py_TYPE(self)->tp_free((PyObject *)self);
}

// --- Solve method ---
PyObject *HSSSolver_Solve(HSSSolverObject *self, PyObject *args) {
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

    
    HSSParametersObject *params_wrapper = (HSSParametersObject *)py_params_obj;   
    SolverParametersObject *base_params_wrapper = (SolverParametersObject *)py_params_obj;   

    // pointers validation
    if (!self->super.solver || !dataset_wrapper->data_set || !params_wrapper->base.params) {
        PyErr_SetString(PyExc_RuntimeError, "Internal C++ object or parameters were not initialized.");
        return NULL;
    }

    moda::HSSResult* result_ptr = NULL;
    moda::HSSParameters* params =  new moda::HSSParameters(moda::HSSParameters::ReferencePointCalculationStyle::zeroone, moda::HSSParameters::ReferencePointCalculationStyle::zeroone);
    try {

        // fprintf(stderr, "DEBUG: params pointer: %p\n", (void*)params_wrapper->base.params);
        // fflush(stderr); 
        params->Strategy = static_cast<moda::HSSParameters*>(params_wrapper->base.params)->Strategy;
        params->StoppingCriteria = static_cast<moda::HSSParameters*>(params_wrapper->base.params)->StoppingCriteria;
        params->StoppingSubsetSize = static_cast<moda::HSSParameters*>(params_wrapper->base.params)->StoppingSubsetSize;
        params->StoppingTime = static_cast<moda::HSSParameters*>(params_wrapper->base.params)->StoppingTime;
        // params->WorseReferencePointCalculationStyle = static_cast<moda::SolverParameters*>(base_params_wrapper->params)->WorseReferencePointCalculationStyle;
        // params->BetterReferencePointCalculationStyle = static_cast<moda::SolverParameters*>(base_params_wrapper->params)->BetterReferencePointCalculationStyle;
        // fprintf(stderr, "DEBUG: solver strategy: %d\n", params->Strategy);
        // fprintf(stderr, "DEBUG: solver criteria: %d\n", params->StoppingCriteria);
        // fprintf(stderr, "DEBUG: solver subsetsize: %d\n", params->StoppingSubsetSize);
        // fprintf(stderr, "DEBUG: solver stoptime: %d\n", params->StoppingTime);
        
        //static cast
        moda::HSSSolver* hss_ptr = static_cast<moda::HSSSolver*>(self->super.solver);
        // fprintf(stderr, "DEBUG: solver pointer: %p\n", (void*)hss_ptr);
        // fprintf(stderr, "DEBUG: dataset pointer: %p\n", (void*)dataset_wrapper->data_set);
        // fprintf(stderr, "DEBUG: number of points: %d\n", (int)dataset_wrapper->data_set->points.size());

        // fflush(stderr);
        // calling the method
        result_ptr = hss_ptr->Solve(
            dataset_wrapper->data_set, 
            *params
        );
        // fprintf(stderr, "DEBUG: results pointer: %p\n", (void*)result_ptr);
        // fprintf(stderr, "DEBUG: results size: %d\n", (int)result_ptr->selectedPoints.size());
        // fflush(stderr);
        delete params;
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }


    // ---- convert vector<int> to Python list ----
    const std::vector<int>& pts = result_ptr->selectedPoints;

    PyObject* pyList = PyList_New(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        PyObject* val = PyLong_FromLong(pts[i]);
        PyList_SetItem(pyList, i, val); 
    }

    // Create a tuple of size 2
    PyObject* pyTuple = PyTuple_New(2);

    PyTuple_SetItem(pyTuple, 0, pyList);
    // PyTuple_SetItem(pyTuple, 0, PyFloat_FromDouble(result_ptr->HyperVolume));
    PyTuple_SetItem(pyTuple, 1, PyFloat_FromDouble(result_ptr->HyperVolume));
    delete result_ptr;
    return pyTuple;
}

// methods
PyMethodDef HSSSolver_methods[] = {
    {"Solve", (PyCFunction)HSSSolver_Solve, METH_VARARGS,
     "Solves the optimization problem using HSS algorithm."},
    {NULL}  /* Sentinel */
};

PyTypeObject HSSSolverType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "moda.HSSSolver",             /* tp_name */
    sizeof(HSSSolverObject),      /* tp_basicsize */
    0, 
    (destructor)HSSSolver_dealloc, /* tp_dealloc (dedicated dealloc) */
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
    "Solver implementation based on HSSSolver.", /* tp_doc */
    0, /* tp_traverse */
    0, /* tp_clear */
    0, /* tp_richcompare: For comparisons (<, >, ==) */
    0, /* tp_weaklistoffset */
    0, /* tp_iter */
    0, /* tp_iternext */
    HSSSolver_methods,            /* tp_methods */
    NULL,                          /* tp_members */
    NULL,                          /* tp_getset */
    &SolverType,                   /* tp_base: derived from SolverType */
    0, /* tp_dict */
    0, /* tp_descr_get */
    0, /* tp_descr_set */
    0, /* tp_dictoffset */
    (initproc)HSSSolver_init,     /* tp_init */
    0, 
    (newfunc)HSSSolver_new,       /* tp_new */
};


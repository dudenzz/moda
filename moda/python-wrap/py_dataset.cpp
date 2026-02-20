#pragma once

#define PY_ARRAY_UNIQUE_SYMBOL moda_ARRAY_API
#define NO_IMPORT_ARRAY // Use this in all files EXCEPT the one where you call import_array()
#include <numpy/arrayobject.h>
#include "moda_types.h"


#include "../DataSet.h"
#include "../Point.h"
#include <Python.h>
#include "moda_types.h"

#if DTypeN == 2
#define NPY_DTYPEN NPY_DOUBLE
#else
#define NPY_DTYPEN NPY_FLOAT
#endif

//Enum type definition for optimization type
PyTypeObject OptimizationTypeType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "moda.OptimizationType",
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 17 zer dla tp_basicsize do tp_iternext
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,                                   // tp_flags
    "Optimization type - minimization, maximization"
};


//Enum initializations
int init_OptimizationType(PyObject *m) {
    PyObject *enum_type_obj = (PyObject *)&OptimizationTypeType;
    if (PyType_Ready(&OptimizationTypeType) < 0) {
            return -1;
        }
    // 1. We put the enum into the module
    Py_INCREF(enum_type_obj);
    if (PyModule_AddObject(m, "OptimizationType", enum_type_obj) < 0) {
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
    ADD_ENUM_CONST_TO_DICT("minimization", moda::DataSet::OptimizationType::minimization);
    ADD_ENUM_CONST_TO_DICT("maximization", moda::DataSet::OptimizationType::maximization);


    #undef ADD_ENUM_CONST_TO_DICT
    
    return 0;
}

void DataSet_dealloc(DataSetObject *self) {
    if (self->data_set) {
        // Calling the C++ destructor ensures all contained Point* are deleted, 
        // nadir and ideal are deleted, and vectors are cleared.
        delete self->data_set; 
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

int DataSet_init(DataSetObject *self, PyObject *args, PyObject *kwds) {
    PyObject *first_arg = NULL;
    
    // Obsługujemy 0 lub 1 argument
    if (!PyArg_ParseTuple(args, "|O", &first_arg)) {
        return -1; // Błąd parsowania
    }

    try {
        if (first_arg == NULL) {
            // SCENARIUSZ 1: DataSet() - Konstruktor domyślny
            self->data_set = new moda::DataSet();
            
        } else if (PyUnicode_Check(first_arg)) {
            // SCENARIUSZ 2: DataSet(filename: str) - Plik
            const char *filename_c = PyUnicode_AsUTF8(first_arg);
            if (!filename_c) return -1; // Błąd konwersji
            
            self->data_set = moda::DataSet::LoadFromFilename(std::string(filename_c));
            
        } else if (PyArray_Check(first_arg)) {
            // SCENARIUSZ 3: DataSet(data: numpy.ndarray) - Tablica NumPy
            
            PyArrayObject *data_array = (PyArrayObject *)first_arg;
            
            // Walidacja 1: Wymiary (musi być tablica 2D)
            if (PyArray_NDIM(data_array) != 2) {
                PyErr_SetString(PyExc_ValueError, "Input data must be a 2D NumPy array (N_points x N_objectives).");
                return -1;
            }

            // Walidacja 2: Typ danych (musi pasować do DType)
            // if (PyArray_TYPE(data_array) != NPY_DTYPEN) {
            //     PyErr_SetString(PyExc_TypeError, "NumPy array dtype must match the C++ DType (float64 or float32).");
            //     return -1;
            // }
            
            Py_ssize_t n_points = PyArray_DIM(data_array, 0);
            Py_ssize_t n_objectives = PyArray_DIM(data_array, 1);

            if (n_objectives > MAXOBJECTIVES) {
                PyErr_Format(PyExc_ValueError, 
                             "Number of objectives (%zd) exceeds the static MAXOBJECTIVES (%d).", 
                             n_objectives, MAXOBJECTIVES);
                return -1;
            }
            
            // 3. Utworzenie obiektu DataSet i załadowanie danych
            self->data_set = new moda::DataSet((int)n_objectives);

            DType *row_ptr = (DType *)PyArray_DATA(data_array);

            for (Py_ssize_t i = 0; i < n_points; ++i) {
                // Utworzenie nowego obiektu Point dla każdego wiersza
                moda::Point *p = new moda::Point();
                
                // Kopiowanie danych: row_ptr wskazuje na początek bieżącego wiersza
                std::copy(row_ptr, 
                          row_ptr + n_objectives, 
                          p->ObjectiveValues);
                
                // for(int j = 0;j<(int)n_objectives; j++)
                // {
                //     std::cout << row_ptr[i*(int)n_objectives + j] << std::endl;
                //     p->ObjectiveValues[j] = (double)row_ptr[i*(int)n_objectives + j];
                // }
                // Ustawienie faktycznej liczby celów
                p->NumberOfObjectives = (int)n_objectives;
                // Dodanie punktu do DataSet (DataSet::add dba o resztę)
                self->data_set->add(p); 
                
                // Przejście do następnego wiersza (zakładamy C-ordering)
                row_ptr += n_objectives; 
            }
            
            // Sprawdzenie i aktualizacja parametrów DataSet po załadowaniu
            self->data_set->setNumberOfPoints((int)n_points);
            self->data_set->setDimensionality((int)n_objectives);


        } else {
            // Nieznany typ argumentu
            PyErr_SetString(PyExc_TypeError, 
                            "DataSet constructor accepts no arguments, a filename (str), or a 2D numpy.ndarray.");
            return -1;
        }

    } catch (const std::runtime_error &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return -1;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unknown error during C++ DataSet initialization.");
        return -1;
    }
    
    return 0; // Sukces
}

Py_ssize_t DataSet_len(DataSetObject *self) {
    return self->data_set->points.size(); 
}

extern PyTypeObject PointType; // Must be declared globally

PyObject *DataSet_subscript(DataSetObject *self, PyObject *key) {
    Py_ssize_t index;
    // ... (Index validation and normalization logic, similar to Point_ass_item) ...
    
    // 1. Retrieve the C++ Point*
    moda::Point *cpp_point = self->data_set->points.at(index);
    if (!cpp_point) {
        // Should not happen if C++ code is stable
        PyErr_SetString(PyExc_ValueError, "Internal Point pointer is NULL.");
        return NULL;
    }
    
    // 2. Create a new Python PointObject
    PointObject *py_point = (PointObject *)PyObject_CallObject((PyObject *)&PointType, NULL);
    if (!py_point) return NULL;
    
    // 3. Set the C++ pointer inside the Python wrapper.
    // NOTE: Because the Point* is MANAGED by DataSet, we must copy it 
    // to give the Python object ownership, or decide that the Python object
    // is just a temporary *reference* (more complex memory model).
    // For simplicity, we create a copy:
    delete py_point->point; // Delete the default Point created by PointType init
    py_point->point = new moda::Point(*cpp_point); // Copy the C++ Point data
    
    // 4. Return the new Python wrapper
    return (PyObject *)py_point;
}
PyObject* DataSet_add(DataSetObject *self, PyObject *args) {
    PyObject *py_point_obj = NULL;
    PointObject *point_wrapper;
    moda::Point *point_to_add;

    // Parse
    if (!PyArg_ParseTuple(args, "O", &py_point_obj)) {
        return NULL; // PyArg_ParseTuple ustawi błąd
    }

    // Validate
    if (!PyObject_TypeCheck(py_point_obj, &PointType)) {
        PyErr_SetString(PyExc_TypeError, 
                        "Argument must be an instance of moda.Point.");
        return NULL;
    }

    point_wrapper = (PointObject *)py_point_obj;

    // --- Ownership Transfer ---
    point_to_add = point_wrapper->point;
    

    // --- Call 
    if (!point_to_add) {
         PyErr_SetString(PyExc_ValueError, "Internal C++ Point pointer was NULL.");
         return NULL;
    }
    
    try {
        bool success = self->data_set->add(point_to_add);
        
        // Python booleans
        if (success) {
            Py_RETURN_TRUE;
        } else {
            Py_RETURN_FALSE;
        }
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unknown error during DataSet::add.");
        return NULL;
    }
}
PyObject* DataSet_normalize(DataSetObject *self, PyObject *Py_UNUSED(ignored)) {


    try {
        // Call
        self->data_set->normalize();

    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unknown error during DataSet::normalize.");
        return NULL;
    }
// it's a procedure, does not have any return value
    Py_RETURN_NONE;
}

PyObject* DataSet_reverse(DataSetObject *self, PyObject *Py_UNUSED(ignored)) {


    try {
        
        self->data_set->reverseObjectives();

    } catch (const std::exception &e) {
       
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unknown error during DataSet::reverseObjectives.");
        return NULL;
    }

    // it's a procedure, does not have any return value
    Py_RETURN_NONE;
}

PyObject* DataSet_get_ideal(DataSetObject *self, PyObject *Py_UNUSED(ignored)) {
    moda::Point *cpp_ideal_point;
    PointObject *py_point_wrapper = NULL;

    try {
        // Call
        cpp_ideal_point = self->data_set->getIdeal();
        
        if (!cpp_ideal_point) {
            Py_RETURN_NONE; 
        }

        // Copy
        py_point_wrapper = (PointObject *)PyObject_CallObject((PyObject *)&PointType, NULL);
        if (!py_point_wrapper) {
            return NULL; 
        }

        // Check if default constructor was called, if so delete the point to prevent mem-leaks
        if (py_point_wrapper->point) {
            delete py_point_wrapper->point;
        }

        // Copy a new point to a wrapped object
        py_point_wrapper->point = new moda::Point(*cpp_ideal_point);

        // return
        return (PyObject *)py_point_wrapper;

    } catch (const std::exception &e) {
        // DECREF handling 
        if (py_point_wrapper) {
            Py_DECREF(py_point_wrapper);
        }
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    } catch (...) {
        if (py_point_wrapper) {
            Py_DECREF(py_point_wrapper);
        }
        PyErr_SetString(PyExc_RuntimeError, "Unknown error during DataSet::getIdeal.");
        return NULL;
    }
}
PyObject* DataSet_str(DataSetObject *self) {
    std::string cpp_string;
    
    // Check if the object exists
    if (!self->data_set) {
        return PyUnicode_FromString("<moda.DataSet object (uninitialized C++ pointer)>");
    }

    try {
        // Call
        cpp_string = self->data_set->to_string();

        // Convert and return
        return PyUnicode_FromStringAndSize(cpp_string.c_str(), cpp_string.length());

    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unknown error during DataSet::to_string().");
        return NULL;
    }
}
// --- Protocols ---
PyMappingMethods DataSet_as_mapping = {
    (lenfunc)DataSet_len,          // mp_length
    (binaryfunc)DataSet_subscript, // mp_subscript (ds[i])
    (objobjargproc)NULL            // We will use a dedicated method for set (ds[i] = p)
};

PyMethodDef DataSet_methods[] = {
    // Expose all public methods here, e.g.:
    {"add", (PyCFunction)DataSet_add, METH_VARARGS, "Adds a Point to the DataSet."},
    {"normalize", (PyCFunction)DataSet_normalize, METH_NOARGS, "Normalizes the DataSet."},
    {"get_ideal", (PyCFunction)DataSet_get_ideal, METH_NOARGS, "Returns the calculated ideal point."},
    {"reverse", (PyCFunction)DataSet_reverse, METH_NOARGS, "Reverses objective values."},
    {NULL}  // Sentinel
};

// GETTER
PyObject* DataSet_get_typeOfOptimization(DataSetObject *self, void *closure) {
    if (!self->data_set) {
        PyErr_SetString(PyExc_RuntimeError, "DataSet pointer is NULL");
        return NULL;
    }
    // Convert C++ Enum to Python Integer
    return PyLong_FromLong((long)self->data_set->typeOfOptimization);
}

// SETTER
int DataSet_set_typeOfOptimization(DataSetObject *self, PyObject *value, void *closure) {
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete typeOfOptimization attribute");
        return -1;
    }

    if (!PyLong_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "typeOfOptimization must be an integer (OptimizationType)");
        return -1;
    }

    long val = PyLong_AsLong(value);
    
    // Validation: Ensure the value is within the Enum range (0 or 1)
    if (val < 0 || val > 1) {
        PyErr_SetString(PyExc_ValueError, "Invalid OptimizationType value");
        return -1;
    }

    if (self->data_set) {
        self->data_set->typeOfOptimization = (moda::DataSet::OptimizationType)val;
    }
    return 0;
}
static PyGetSetDef DataSet_getseters[] = {
    {"typeOfOptimization", 
     (getter)DataSet_get_typeOfOptimization, 
     (setter)DataSet_set_typeOfOptimization, 
     "Optimization type (0: minimization, 1: maximization)", 
     NULL},
    {NULL}  /* Sentinel */
};
PyTypeObject DataSetType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "moda.Dataset",              /* tp_name: Name of the type (module.class) */
    sizeof(DataSetObject),       /* tp_basicsize: Size of the structure that holds the C++ object */
    0,                         /* tp_itemsize */
    (destructor)DataSet_dealloc, /* tp_dealloc: Called when object is garbage collected */
    0,                         /* tp_print (Deprecated) */
    0,                         /* tp_getattr (Deprecated) */
    0,                         /* tp_setattr (Deprecated) */
    0,                         /* tp_compare (Deprecated) */
    0,                         /* tp_repr (Optional: string representation) */
    0,                         /* tp_as_number: Used for arithmetic operators (+, -, *, etc.) */
    0,                         /* tp_as_sequence: Used for sequence protocols (tuple, list) */
    &DataSet_as_mapping,         /* tp_as_mapping: Used for indexing (p[i]) */
    0,                         /* tp_hash */
    0,                         /* tp_call */
    (reprfunc)DataSet_str,                         /* tp_str: Used for str(p) */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | 
    Py_TPFLAGS_BASETYPE,       /* tp_flags: Standard flags */
    "C++ moda::DataSet for storing multiple points.", /* tp_doc: Class documentation */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare: For comparisons (<, >, ==) */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    DataSet_methods,             /* tp_methods: The static/instance methods table */
    0,                         /* tp_members */
    DataSet_getseters,          /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)DataSet_init,      /* tp_init: Called after allocation (initializes C++ object) */
    0,                         /* tp_alloc */
    (newfunc)PyType_GenericNew, /* tp_new: Called to allocate the memory structure */
};
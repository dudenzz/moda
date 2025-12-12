#pragma once
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "../Point.h"
#include <Python.h>
#include <stdexcept>
#include <cassert>
#include <algorithm> // For std::copy



// Structure for the Python Point object
typedef struct {
    PyObject_HEAD             // Required boilerplate for Python objects
    moda::Point *point;       // Pointer to the actual C++ Point object
} PointObject;

static void Point_dealloc(PointObject *self) {
    if (self->point) {
        // Clean up the dynamically allocated C++ object
        delete self->point;
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int Point_init(PointObject *self, PyObject *args, PyObject *kwds) {
    PyObject *input_array = NULL;
    
    // --- 1. Allocate C++ object ---
    try {
        // Allocate the C++ object on the heap
        self->point = new moda::Point();
    } catch (const std::bad_alloc& e) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate C++ moda::Point.");
        return -1;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unknown error during C++ Point allocation.");
        return -1;
    }

    // --- 2. Parse Arguments ---
    // The init signature expects one optional argument: the NumPy array
    if (!PyArg_ParseTuple(args, "|O", &input_array)) {
        // If parsing failed (wrong number/type of arguments, but it should succeed
        // because we handle the optional 'O'), clean up the allocated C++ object.
        delete self->point;
        self->point = NULL;
        return -1;
    }

    // If no argument was provided, use the default constructor and finish.
    if (input_array == NULL) {
        // The moda::Point() constructor has already been called implicitly
        return 0;
    }
    
    // --- 3. Validate and Prepare NumPy Array ---
    // Convert the input Python object to a NumPy array in C_CONTIGUOUS, DType format (e.g., NPY_DOUBLE)
    PyArrayObject *np_array = (PyArrayObject *)PyArray_FROM_OTF(
        input_array, 
        NPY_DOUBLE, // Use the NumPy type corresponding to moda::DType (e.g., NPY_DOUBLE for double)
        NPY_ARRAY_IN_ARRAY // Flags: ensures C contiguous, readable array
    );

    if (!np_array) {
        // PyArray_FROM_OTF sets the appropriate Python exception
        delete self->point;
        self->point = NULL;
        return -1;
    }

    // --- 4. Check Dimensions ---
    if (PyArray_NDIM(np_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Input array must be 1-dimensional.");
        Py_DECREF(np_array);
        delete self->point;
        self->point = NULL;
        return -1;
    }

    // --- 5. Check Size against MAXOBJECTIVES ---
    Py_ssize_t input_size = PyArray_SIZE(np_array);
    self->point->NumberOfObjectives = input_size;
    if (input_size > MAXOBJECTIVES) {
        char err_msg[128];
        snprintf(err_msg, sizeof(err_msg), 
                 "Input array size (%zd) must be less than maximum size (%d).", 
                 input_size, MAXOBJECTIVES);
        PyErr_SetString(PyExc_ValueError, err_msg);
        Py_DECREF(np_array);
        delete self->point;
        self->point = NULL;
        return -1;
    }

    // --- 6. Copy Data ---
    // PyArray_DATA(np_array) returns a pointer to the raw data buffer.
    DType *data_ptr = (DType *)PyArray_DATA(np_array);
    
    // Copy the data into the C++ Point's fixed array
    std::copy(data_ptr, 
            data_ptr + input_size, 
            // POPRAWKA: Użycie nazwy tablicy jako wskaźnika na pierwszy element
            self->point->ObjectiveValues);

    // --- 7. Clean up NumPy reference ---
    Py_DECREF(np_array);

    return 0; // Success
}

static Py_ssize_t Point_len(PointObject *self) {
    // Return the statically defined size (MAXOBJECTIVES, which we aliased to NUM_OBJECTIVES)
    return self->point->NumberOfObjectives; 
}

static PyObject *Point_subscript(PointObject *self, PyObject *key) {
    Py_ssize_t index;

    // 1. Get the integer index from the Python key
    if (PyIndex_Check(key)) {
        index = PyNumber_AsSsize_t(key, NULL);
        if (index < 0 || index >= Point_len(self)) {
            PyErr_SetString(PyExc_IndexError, "Index out of range");
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Point indices must be integers");
        return NULL;
    }

    // 2. Return the value as a Python float
    return PyFloat_FromDouble(self->point->ObjectiveValues[index]);
}
// You'll also need Point_ass_item for setting values (p[i] = x)

static PyObject *Point_zeroes(PyTypeObject *type, PyObject *args) {
    // 1. Create a new Python PointObject instance
    PointObject *new_py_point = (PointObject *)PyObject_CallObject((PyObject *)type, NULL);
    if (!new_py_point) return NULL;

    // 2. Call the C++ static method and assign the result
    // Since Point::zeroes returns a Point by value, we need a copy.
    *(new_py_point->point) = moda::Point::zeroes(new_py_point->point->NumberOfObjectives);

    return (PyObject *)new_py_point;
}

static PyObject* Point_str(PointObject *self) {
    // We will use a fixed-size buffer for efficiency, assuming MAXOBJECTIVES * 20 characters is enough.
    // 20 characters allows for the number + comma + space.
    const size_t MAX_STR_LEN = (size_t)MAXOBJECTIVES * 20 + 32; // +32 for "Point([])" and safety
    char buffer[MAX_STR_LEN];
    char *ptr = buffer;
    int len;

    // 1. Start the string: "Point(["
    len = snprintf(ptr, MAX_STR_LEN, "Point([");
    if (len < 0 || len >= MAX_STR_LEN) {
        // Handle error: buffer overflow
        PyErr_SetString(PyExc_RuntimeError, "String buffer too small for Point representation.");
        return NULL;
    }
    ptr += len;
    
    // 2. Iterate through objective values and append them
    for (int i = 0; i < self->point->NumberOfObjectives; i++) {
        // Append the current DType value (using %g for general float format)
        len = snprintf(ptr, MAX_STR_LEN - (ptr - buffer), "%g", self->point->ObjectiveValues[i]);
        if (len < 0 || len >= MAX_STR_LEN - (ptr - buffer)) {
            PyErr_SetString(PyExc_RuntimeError, "String buffer overflow during value conversion.");
            return NULL;
        }
        ptr += len;

        // Append ", " if it's not the last element
        if (i < self->point->NumberOfObjectives - 1) {
            len = snprintf(ptr, MAX_STR_LEN - (ptr - buffer), ", ");
            ptr += len;
        }
    }
    
    // 3. Close the string: "])"
    len = snprintf(ptr, MAX_STR_LEN - (ptr - buffer), "])");
    ptr += len;

    // 4. Convert the C string (buffer) to a Python Unicode object
    return PyUnicode_FromString(buffer);
}

static int Point_ass_item(PointObject *self, PyObject *key, PyObject *value) {
    Py_ssize_t index;
    DType new_value; // Typ docelowy w C++

    // --- 1. Sprawdzenie, czy wartość nie jest usuwana ---
    if (value == NULL) {
        // Python traktuje p[i] = None jako próbę usunięcia elementu (del p[i])
        PyErr_SetString(PyExc_TypeError, "Cannot delete elements from Point (fixed-size array)");
        return -1;
    }

    // --- 2. Analiza klucza (indeksu) i Sprawdzenie Zakresu ---
    
    // Sprawdzenie, czy klucz jest indeksem (liczbą całkowitą)
    if (!PyIndex_Check(key)) {
        PyErr_SetString(PyExc_TypeError, "Point index must be an integer.");
        return -1;
    }
    
    // Konwersja klucza na Py_ssize_t (bezpieczny typ dla indeksów)
    index = PyNumber_AsSsize_t(key, PyExc_IndexError);
    if (index == -1 && PyErr_Occurred()) {
        // Błąd konwersji (np. zbyt duża liczba)
        return -1; 
    }
    
    // Normalizacja indeksu (obsługa ujemnych indeksów z Pythona)
    if (index < 0) {
        index += self->point->NumberOfObjectives; // MAXOBJECTIVES jest stałą
    }

    // Sprawdzenie, czy indeks jest poza zakresem [0, MAXOBJECTIVES - 1]
    if (index < 0 || index >= self->point->NumberOfObjectives) {
        PyErr_SetString(PyExc_IndexError, "Index out of bounds for Point objectives.");
        return -1;
    }

    // --- 3. Analiza wartości ---
    
    // Sprawdzenie, czy wartość jest liczbą zmiennoprzecinkową lub całkowitą
    if (!PyFloat_Check(value) && !PyLong_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Objective value must be a number (float or int).");
        return -1;
    }
    
    // Konwersja wartości na DType (np. double)
    new_value = (DType)PyFloat_AsDouble(value);
    
    // PyFloat_AsDouble zwraca -1.0 w przypadku błędu konwersji, 
    // ale PyErr_Occurred() jest bardziej niezawodne.
    if (PyErr_Occurred()) {
        // Jeśli konwersja się nie powiodła (np. nie mieści się w double)
        return -1;
    }

    // --- 4. Przypisywanie ---
    
    // Wykonanie bezpiecznego przypisania do statycznej tablicy C++
    self->point->ObjectiveValues[index] = new_value;

    return 0; // Sukces
}

// --- C-API Getter Function (for PyGetSetDef) ---
static PyObject* Point_get_num_objectives(PointObject *self, void *closure) {
    // Call the C++ getter
    return PyLong_FromLong(self->point->NumberOfObjectives);
}

// --- C-API Setter Function (for PyGetSetDef) ---
static int Point_set_num_objectives(PointObject *self, PyObject *value, void *closure) {
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the number of objectives attribute");
        return -1;
    }
    
    if (!PyLong_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Number of objectives must be an integer");
        return -1;
    }
    
    long n = PyLong_AsLong(value);
    
    // Call the C++ setter (which includes the safety check against MAXOBJECTIVES)
    self->point->NumberOfObjectives = (int)n;
    
    // Check if the C++ setter clamped the value (optional, but good practice)
    if (self->point->NumberOfObjectives != (int)n) {
        PyErr_WarnEx(PyExc_UserWarning, "Attempted size exceeds MAXOBJECTIVES; value clamped to maximum.", 1);
    }
    
    return 0; // Success
}
static PyMappingMethods Point_as_mapping = {
    (lenfunc)Point_len,          // mp_length
    (binaryfunc)Point_subscript, // mp_subscript (p[i])
    (objobjargproc)Point_ass_item // mp_ass_subscript (p[i] = x)
};

static PyMethodDef Point_methods[] = {
    {"zeroes", (PyCFunction)Point_zeroes, METH_STATIC | METH_VARARGS, "Returns a Point initialized to zero."},
    // Add other methods (Distance, Compare, etc.) here
    {NULL}  // Sentinel
};

static PyGetSetDef Point_getsetters[] = {
    {"NumberOfObjectives", 
     (getter)Point_get_num_objectives, 
     (setter)Point_set_num_objectives,
     "The number of active objective values (must be <= MAXOBJECTIVES).", 
     NULL},
    {NULL}  // Sentinel
};

static PyTypeObject PointType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "moda.Point",              /* tp_name: Name of the type (module.class) */
    sizeof(PointObject),       /* tp_basicsize: Size of the structure that holds the C++ object */
    0,                         /* tp_itemsize */
    (destructor)Point_dealloc, /* tp_dealloc: Called when object is garbage collected */
    0,                         /* tp_print (Deprecated) */
    0,                         /* tp_getattr (Deprecated) */
    0,                         /* tp_setattr (Deprecated) */
    0,                         /* tp_compare (Deprecated) */
    0,                         /* tp_repr (Optional: string representation) */
    0,                         /* tp_as_number: Used for arithmetic operators (+, -, *, etc.) */
    0,                         /* tp_as_sequence: Used for sequence protocols (tuple, list) */
    &Point_as_mapping,         /* tp_as_mapping: Used for indexing (p[i]) */
    0,                         /* tp_hash */
    0,                         /* tp_call */
    (reprfunc)Point_str,                         /* tp_str: Used for str(p) */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | 
    Py_TPFLAGS_BASETYPE,       /* tp_flags: Standard flags */
    "C++ moda::Point object for objective values.", /* tp_doc: Class documentation */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare: For comparisons (<, >, ==) */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    Point_methods,             /* tp_methods: The static/instance methods table */
    0,                         /* tp_members */
    Point_getsetters,           /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)Point_init,      /* tp_init: Called after allocation (initializes C++ object) */
    0,                         /* tp_alloc */
    (newfunc)PyType_GenericNew, /* tp_new: Called to allocate the memory structure */
};

static PyObject* Point_create_copy(moda::Point* cpp_point) {
    // 1. Walidacja wskaźnika
    if (!cpp_point) {
        Py_RETURN_NONE;
    }
    
    PointObject *py_point_wrapper = NULL;

    try {
        // 2. Alokacja nowego obiektu Python (PointObject)
        // PyObject_CallObject wywołuje tp_new i tp_init PointType.
        py_point_wrapper = (PointObject *)PyObject_CallObject((PyObject *)&PointType, NULL);
        if (!py_point_wrapper) {
            // PyErr ustawiony przez tp_new/tp_init
            return NULL; 
        }

        // 3. Zarządzanie wewnętrznym wskaźnikiem C++
        // Jeśli domyślny konstruktor PointObject alokuje moda::Point*, musimy go usunąć.
        if (py_point_wrapper->point) {
            delete py_point_wrapper->point;
        }

        // 4. Tworzenie KOPII obiektu C++ i przypisanie własności do Pythona
        py_point_wrapper->point = new moda::Point(*cpp_point);

    } catch (const std::bad_alloc &e) {
        // Błąd alokacji pamięci dla kopii C++
        if (py_point_wrapper) {
            Py_DECREF(py_point_wrapper); // Zwolnienie obiektu Python
        }
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for C++ Point copy.");
        return NULL;
    } catch (const std::exception &e) {
        // Inne błędy C++
        if (py_point_wrapper) {
            Py_DECREF(py_point_wrapper);
        }
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }

    // 5. Zwrot nowego obiektu (licznika referencji = 1)
    return (PyObject *)py_point_wrapper;
}
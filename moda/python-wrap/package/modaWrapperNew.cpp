#include <Python.h>
#include "numpy/arrayobject.h"
#include <cstdio>
#include "Result.h"
#include "Point.h"
#include "DataSet.h"
#include "DataSetParameters.h"
#include "iostream"

using namespace moda;
#pragma region results
#pragma region Result_Wrappers

// Base structure for all results
typedef struct {
    PyObject_HEAD
    Result* result; // Pointer to C++ object
} PyResultObject;

// HypervolumeResult
typedef struct {
    PyResultObject base;
    HypervolumeResult* result;  // Pointer to C++ HypervolumeResult
} PyHypervolumeResultObject;

// R2Result
typedef struct {
    PyResultObject base;
    R2Result* result;  // Pointer to C++ R2Result
} PyR2ResultObject;

// HSSResult (Hypervolume Subset Selection Result)
typedef struct {
    PyResultObject base;
    HSSResult* result;  // Pointer to C++ HSSResult
} PyHSSResultObject;

// BoundedResult (Base for estimation/bounding)
typedef struct {
    PyResultObject base;
    BoundedResult* result;  // Pointer to C++ BoundedResult
} PyBoundedResultObject;

// QEHCResult (Contribution analysis)
typedef struct {
    PyResultObject base;
    QEHCResult* result;  // Pointer to C++ QEHCResult
} PyQEHCResultObject;

// QHV_BQResult (Inherits from BoundedResult)
typedef struct {
    PyBoundedResultObject base;
    QHV_BQResult* result;  // Pointer to C++ QHV_BQResult
} PyQHV_BQResultObject;

// QHV_BRResult (Inherits from BoundedResult)
typedef struct {
    PyBoundedResultObject base;
    QHV_BRResult* result;  // Pointer to C++ QHV_BRResult
} PyQHV_BRResultObject;

// MCHVResult (Inherits from BoundedResult)
typedef struct {
    PyBoundedResultObject base;
    MCHVResult* result;  // Pointer to C++ MCHVResult
} PyMCHVResultObject;

// DBHVEResult (Inherits from BoundedResult)
typedef struct {
    PyBoundedResultObject base;
    DBHVEResult* result;  // Pointer to C++ DBHVEResult
} PyDBHVEResultObject;

#pragma endregion
#pragma region Result functions
// Deallocate function for Python object
static void PyResult_dealloc(PyResultObject* self) {
    delete self->result;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

// Constructor
static PyObject* PyResult_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    PyResultObject* self;
    self = (PyResultObject*)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->result = new Result();  // Create C++ object
    }
    return (PyObject*)self;
}

// Get elapsed time
static PyObject* PyResult_get_elapsed_time(PyResultObject* self, void* closure) {
    return PyLong_FromLong(self->result->ElapsedTime);
}

// Set elapsed time
static int PyResult_set_elapsed_time(PyResultObject* self, PyObject* value, void* closure) {
    if (!PyLong_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "ElapsedTime must be an integer");
        return -1;
    }
    self->result->ElapsedTime = PyLong_AsLong(value);
    return 0;
}

// Get final result flag
static PyObject* PyResult_get_final_result(PyResultObject* self, void* closure) {
    return PyBool_FromLong(self->result->FinalResult);
}

// Set final result flag
static int PyResult_set_final_result(PyResultObject* self, PyObject* value, void* closure) {
    if (!PyBool_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "FinalResult must be a boolean");
        return -1;
    }
    self->result->FinalResult = PyObject_IsTrue(value);
    return 0;
}

// Get result type
static PyObject* PyResult_get_result_type(PyResultObject* self, void* closure) {
    return PyLong_FromLong(self->result->type);
}

// Set result type
static int PyResult_set_result_type(PyResultObject* self, PyObject* value, void* closure) {
    if (!PyLong_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Result Type must be integer");
        return -1;
    }
    self->result->type = (Result::ResultType)PyLong_AsLong(value);
    return 0;
}

// Define properties
static PyGetSetDef PyResult_getsetters[] = {
    {"elapsed_time", (getter)PyResult_get_elapsed_time, (setter)PyResult_set_elapsed_time, "Elapsed Time", NULL},
    {"final_result", (getter)PyResult_get_final_result, (setter)PyResult_set_final_result, "Final Result", NULL},
    {"result_type", (getter)PyResult_get_result_type, (setter)PyResult_set_result_type, "Result Type", NULL},
    {NULL}  // Sentinel
};

// Define type methods
static PyTypeObject PyResultType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "moda.Result",                // tp_name
    sizeof(PyResultObject),      // tp_basicsize
    0,                           // tp_itemsize
    (destructor)PyResult_dealloc, // tp_dealloc
    0,                           // tp_print
    0,                           // tp_getattr
    0,                           // tp_setattr
    0,                           // tp_reserved
    0,                           // tp_repr
    0,                           // tp_as_number
    0,                           // tp_as_sequence
    0,                           // tp_as_mapping
    0,                           // tp_hash  
    0,                           // tp_call
    0,                           // tp_str
    0,                           // tp_getattro
    0,                           // tp_setattro
    0,                           // tp_as_buffer
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, // tp_flags
    "Result objects",           // tp_doc
    0,                           // tp_traverse
    0,                           // tp_clear
    0,                           // tp_richcompare
    0,                           // tp_weaklistoffset
    0,                           // tp_iter
    0,                           // tp_iternext
    0,                           // tp_methods
    0,                           // tp_members
    PyResult_getsetters,        // tp_getset
    0,                           // tp_base
    0,                           // tp_dict
    0,                           // tp_descr_get
    0,                           // tp_descr_set
    0,                           // tp_dictoffset
    0,                           // tp_init
    0,                           // tp_alloc
    PyResult_new,               // tp_new
};

// Module methods
static PyMethodDef module_methods[] = {
    {NULL, NULL, 0, NULL}  // Sentinel
};

#pragma endregion
#pragma region HypervolumeResult functions
// Constructor for HypervolumeResult
static PyObject* PyHypervolumeResult_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    PyHypervolumeResultObject* self;
    self = (PyHypervolumeResultObject*)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->result = new HypervolumeResult();
        self->base.result = self->result;  // Point base class to the subclass object
    }
    return (PyObject*)self;
}

// Getter for volume in HypervolumeResult
static PyObject* PyHypervolumeResult_get_volume(PyHypervolumeResultObject* self, void* closure) {
    return PyFloat_FromDouble(self->result->HyperVolume);
}

// Define getters and setters for HypervolumeResult
static PyGetSetDef PyHypervolumeResult_getsetters[] = {
    {"volume", (getter)PyHypervolumeResult_get_volume, NULL, "Hypervolume", NULL},
    {NULL}  // Sentinel
};

// Define the type for HypervolumeResult
static PyTypeObject PyHypervolumeResultType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "moda.HypervolumeResult",      // tp_name
    sizeof(PyHypervolumeResultObject), // tp_basicsize
    0,                           // tp_itemsize
    (destructor)PyResult_dealloc, // tp_dealloc
    0,                           // tp_print
    0,                           // tp_getattr
    0,                           // tp_setattr
    0,                           // tp_reserved
    0,                           // tp_repr
    0,                           // tp_as_number
    0,                           // tp_as_sequence
    0,                           // tp_as_mapping
    0,                           // tp_hash
    0,                           // tp_call
    0,                           // tp_str
    0,                           // tp_getattro
    0,                           // tp_setattro
    0,                           // tp_as_buffer
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, // tp_flags
    "HypervolumeResult objects", // tp_doc
    0,                           // tp_traverse
    0,                           // tp_clear
    0,                           // tp_richcompare
    0,                           // tp_weaklistoffset
    0,                           // tp_iter
    0,                           // tp_iternext
    0,                           // tp_methods
    0,                           // tp_members
    PyHypervolumeResult_getsetters, // tp_getset
    (PyTypeObject*)&PyResultType,  // tp_base
    0,                           // tp_dict
    0,                           // tp_descr_get
    0,                           // tp_descr_set
    0,                           // tp_dictoffset
    0,                           // tp_init
    0,                           // tp_alloc
    PyHypervolumeResult_new,     // tp_new
};



// Module methods
static PyMethodDef HypervolumeResult_methods[] = {
    {NULL, NULL, 0, NULL}  // Sentinel
};



#pragma endregion
#pragma endregion

#pragma region point
typedef struct {
    PyObject_HEAD;
    Point * ptrObj;
} PyPoint;
#pragma region functions
// ----------------------- Initializer ----------------------------------
static int PyPoint_init(PyPoint *self, PyObject *args, PyObject *kwds)
{
    PyObject *arg,  *out = NULL;
    PyArrayObject *arr;
    if(!PyArg_ParseTuple(args,"|O", &out))
    {
        self->ptrObj = new Point();
        return 0;
    }
    else
    {
        if(out == NULL)
        {
            
            self->ptrObj = new Point();
            return 0; 
        }
        else
        {
            arr = (PyArrayObject*)PyArray_FROM_OTF(out, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
            int ndim =  PyArray_NDIM(arr);
            if(ndim != 1)
            {
                std::cout << "Data object has wrong dimensionality. Expected a 1-D array, recieved " << ndim<< "-D array. Point was created, but it is empty." << std::endl;
                Py_DECREF(arr);
                self->ptrObj = new Point();
                return 0;
            }
            int size = PyArray_DIM(arr,0);
            if(size > 12)
            {   
                std::cout << "Data object is too large. Expected size of a data object is <= 12. Recieved size :" << size << ". Point was created, but it is empty." << std::endl;
                Py_DECREF(arr);
                self->ptrObj = new Point();
                return 0;
            }
            double* values = (double*) PyArray_DATA(arr);
            self->ptrObj = new Point(size);
            self->ptrObj->NumberOfObjectives = size;
            for(int i = 0; i<size; i++)
            {
                self->ptrObj->ObjectiveValues[i] = values[i];
            }
            Py_DECREF(arr);
            return 0; 

        }
    }
}

static PyObject* PyPoint_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    PyPoint* self = (PyPoint*)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->ptrObj = new Point();  // Create a new C++ object
    }
    return (PyObject*)self;
}
// --------------------------------- Deallocator --------------------------------------
static void PyPoint_dealloc(PyPoint * self)
{
    delete self->ptrObj;
    Py_TYPE(self)->tp_free(self);
}
//---------------------------------- Type ---------------------------------------------
extern PyTypeObject PyPointType;
// --------------------------------- Point methods ------------------------------------
static PyObject * PyPoint_get(PyPoint* point, PyObject* args)
{
    int i;
    if (! PyArg_ParseTuple(args, "i", &i))
         return Py_False;
    double value = point->ptrObj->ObjectiveValues[i];
    return Py_BuildValue("d",value);
}

static PyObject * PyPoint_set(PyPoint* point, PyObject* args)
{
    int i;
    double d;
    if (! PyArg_ParseTuple(args, "id", &i, &d))
         return Py_False;
    point->ptrObj->ObjectiveValues[i] = d;
    return Py_None;
}

static PyObject * PyPoint_distance(PyPoint* point, PyObject *args)
{
    PyPoint *compared, *ideal, *nadir;
    if (! PyArg_ParseTuple(args, "OOO", &compared, &ideal, &nadir))
         return Py_False;
    double distance = point->ptrObj->Distance(*compared->ptrObj, *ideal->ptrObj, *nadir->ptrObj);
    return Py_BuildValue("d",distance);
}
static PyObject * PyPoint_ones(PyObject* module, PyObject *args)
{
    int size;
    if (! PyArg_ParseTuple(args, "i", &size))
         return Py_False;
    PyPoint* pointObject = PyObject_New(PyPoint, &PyPointType);
    pointObject->ptrObj = new Point(Point::ones(size));;
    return (PyObject*)pointObject;
}
static PyObject * PyPoint_zeroes(PyObject* module, PyObject *args)
{
    int size;
    if (! PyArg_ParseTuple(args, "i", &size))
         return Py_False;
    PyPoint* pointObject = PyObject_New(PyPoint, &PyPointType);
    pointObject->ptrObj = new Point(Point::ones(size));;
    return (PyObject*)pointObject;
}

static PyObject* PyPoint_getitem(PyPoint* self, PyObject* key)
{
    if (!PyLong_Check(key))
    {
        PyErr_SetString(PyExc_TypeError, "Index must be an integer");
        return NULL;
    }

    int index = PyLong_AsLong(key);
    try {
        double value = self->ptrObj->get(index);
        return Py_BuildValue("d", value);
    }
    catch (const std::out_of_range&) {
        PyErr_SetString(PyExc_IndexError, "Index out of range");
        return NULL;
    }
}

static int PyPoint_setitem(PyPoint* self, PyObject* key, PyObject* value)
{
    if (!PyLong_Check(key) || !PyFloat_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "Index must be an integer and value must be a float");
        return -1;
    }

    int index = PyLong_AsLong(key);
    double val = PyFloat_AsDouble(value);
    
    try {
        self->ptrObj->operator[](index) = val;
        return 0;
    }
    catch (const std::out_of_range&) {
        PyErr_SetString(PyExc_IndexError, "Index out of range");
        return -1;
    }
}



static PyObject* PyPoint_compare(PyPoint* self, PyObject* args)
{
    PyPoint* otherPointObj;
    int direction;
    // Parse the arguments
    if (!PyArg_ParseTuple(args, "Oi", &otherPointObj, &direction))
        return NULL;

    // Ensure the other object is actually a PyPoint object
    if (!PyObject_TypeCheck(otherPointObj, &PyPointType))
    {
        PyErr_SetString(PyExc_TypeError, "Argument must be a Point object");
        return NULL;
    }

    // Call the C++ Compare method
    bool d = direction ? true : false ;
    ComparisonResult result = self->ptrObj->Compare(*otherPointObj->ptrObj, d);

    // Map the result to a Python object
    switch (result) {
        case _Dominating:
            return Py_BuildValue("s", "Dominating");
        case _Dominated:
            return Py_BuildValue("s", "Dominated");
        case _Nondominated:
            return Py_BuildValue("s", "Nondominated");
        case _EqualSol:
            return Py_BuildValue("s", "EqualSol");
        default:
            return NULL;
    }
}


static PyObject* PyPoint_toString(PyObject* self)
{
    PyPoint* point = (PyPoint*)self;
    std::string strBuilder = "[";
    for(int i = 0; i<point->ptrObj->NumberOfObjectives; i++)
    {
        strBuilder += std::to_string(point->ptrObj->ObjectiveValues[i]) + ",";
    }
    strBuilder = strBuilder.substr(0, strBuilder.length()-2) + "]";
    return PyUnicode_FromFormat(strBuilder.c_str());
}
static PyObject* PyPoint_to_numpy(PyPoint* self, PyObject* new_point)
{
    try {

        std::vector<double> vec;

        for(int i = 0; i< self->ptrObj->NumberOfObjectives; i++)
        {
            vec.push_back(self->ptrObj->ObjectiveValues[i]);
        }


        int length = vec.size();
        npy_intp dims[1];
        dims[0] = self->ptrObj->NumberOfObjectives;
        double* data = new double[length];
        std::copy(vec.begin(), vec.end(), data); 
        PyObject* obj = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, (void*)data);
        return obj;
    }
    catch (const std::out_of_range&) {
        PyErr_SetString(PyExc_IndexError, "Index out of range");
        return Py_False;
    }
}
static PyMethodDef PyPoint_methods[] = {
    { "get",(PyCFunction)PyPoint_get, METH_VARARGS, "get the value of an n-th component from a point" },
    { "set",(PyCFunction)PyPoint_set, METH_VARARGS, "set the value of an n-th component in a point" },
    { "distance",(PyCFunction)PyPoint_distance, METH_VARARGS, "calculate a distance between two points given two reference points" },
    { "ones",(PyCFunction)PyPoint_ones, METH_VARARGS | METH_CLASS, "create a Point, for which all components are equal to 1" },
    { "zeroes",(PyCFunction)PyPoint_zeroes, METH_VARARGS | METH_CLASS, "create a Point, for which all components are equal to 0" },
    { "compare",(PyCFunction)PyPoint_compare, METH_VARARGS, "compare 2 points in the context of domination" },
    { "to_numpy",(PyCFunction)PyPoint_to_numpy, METH_VARARGS, "convert to numpy array" },
    {NULL}  /* Sentinel */
};
// Define the type for Point

PyTypeObject PyPointType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "moda.Point",      // tp_name
    sizeof(PyPoint), // tp_basicsize
    0,                           // tp_itemsize
    (destructor)PyPoint_dealloc, // tp_dealloc
    0,                           // tp_print
    0,                           // tp_getattr
    0,                           // tp_setattr
    0,                           // tp_reserved
    0,                           // tp_repr
    0,                           // tp_as_number
    0,                           // tp_as_sequence
    0,                           // tp_as_mapping
    0,                           // tp_hash
    0,                           // tp_call
    0,                           // tp_str
    0,                           // tp_getattro
    0,                           // tp_setattro
    0,                           // tp_as_buffer
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, // tp_flags
    "Point object", // tp_doc
    0,                           // tp_traverse
    0,                           // tp_clear
    0,                           // tp_richcompare
    0,                           // tp_weaklistoffset
    0,                           // tp_iter
    0,                           // tp_iternext
    PyPoint_methods,                           // tp_methods
    0,                           // tp_members
    0 , // tp_getset
    0,  // tp_base
    0,                           // tp_dict
    0,                           // tp_descr_get
    0,                           // tp_descr_set
    0,                           // tp_dictoffset
    (initproc)PyPoint_init,                           // tp_init
    0,                           // tp_alloc
    PyPoint_new,     // tp_new
};
static PyMappingMethods PyPoint_mapping = {
    (lenfunc) NULL,            // No __len__()
    (binaryfunc) PyPoint_getitem,  // __getitem__()
    (objobjargproc) PyPoint_setitem,  // __setitem__()
};

#pragma endregion
#pragma endregion

#pragma region dataset

#pragma region dataset params
typedef struct {
    PyObject_HEAD
    DataSetParameters* ptrObj;  // Pointer to the actual C++ DataSetParameters object
} PyDataSetParametersObject;
#pragma region functions
  static void PyDataSetParameters_dealloc(PyDataSetParametersObject* self) {
        delete self->ptrObj;  // Clean up the C++ object when the Python object is deleted
        Py_TYPE(self)->tp_free((PyObject*)self);
    }

    static PyObject* PyDataSetParameters_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
        PyDataSetParametersObject* self = (PyDataSetParametersObject*)type->tp_alloc(type, 0);
        if (self != NULL) {
            self->ptrObj = new DataSetParameters();  // Create a new C++ object
        }
        return (PyObject*)self;
    }

    static int PyDataSetParameters_init(PyDataSetParametersObject* self, PyObject* args, PyObject* kwds) {
        const char* name = nullptr;
        int dimensions = 0;
        int nPoints = 0;
        int sampleNumber = 0;
        const char* filename = nullptr;

        static const char* kwlist[] = {"name", "dimensions", "nPoints", "sampleNumber", "filename", nullptr};
        if (!PyArg_ParseTupleAndKeywords(args, kwds, "|siiii", (char**)kwlist, &name, &dimensions, &nPoints, &sampleNumber, &filename)) {
            return -1;
        }

        // Handle constructors
        if (name != nullptr) {
            self->ptrObj = new DataSetParameters(std::string(name), dimensions, nPoints, sampleNumber);
        } else if (filename != nullptr) {
            self->ptrObj = new DataSetParameters(std::string(filename));
        } else {
            self->ptrObj = new DataSetParameters();  // Default constructor
        }

        return 0;
    }

    // Define getter methods for each property
    static PyObject* PyDataSetParameters_get_filename(PyDataSetParametersObject* self, void* closure) {
        return PyUnicode_FromString(self->ptrObj->filename.c_str());
    }

    static PyObject* PyDataSetParameters_get_name(PyDataSetParametersObject* self, void* closure) {
        return PyUnicode_FromString(self->ptrObj->name.c_str());
    }

    static PyObject* PyDataSetParameters_get_NumberOfObjectives(PyDataSetParametersObject* self, void* closure) {
        return PyLong_FromLong(self->ptrObj->NumberOfObjectives);
    }

    static PyObject* PyDataSetParameters_get_nPoints(PyDataSetParametersObject* self, void* closure) {
        return PyLong_FromLong(self->ptrObj->nPoints);
    }

    static PyObject* PyDataSetParameters_get_sampleNumber(PyDataSetParametersObject* self, void* closure) {
        return PyLong_FromLong(self->ptrObj->sampleNumber);
    }



    static PyGetSetDef PyDataSetParameters_getsetters[] = {
        {"filename", (getter)PyDataSetParameters_get_filename, NULL, "Filename", NULL},
        {"name", (getter)PyDataSetParameters_get_name, NULL, "Experiment name", NULL},
        {"NumberOfObjectives", (getter)PyDataSetParameters_get_NumberOfObjectives, NULL, "Number of objectives", NULL},
        {"nPoints", (getter)PyDataSetParameters_get_nPoints, NULL, "Number of points", NULL},
        {"sampleNumber", (getter)PyDataSetParameters_get_sampleNumber, NULL, "Sample number", NULL},
        {NULL}  // Sentinel value
    };

    static PyTypeObject PyDataSetParametersType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "moda.DataSetParameters",      // Name of the Python class
        sizeof(PyDataSetParametersObject),
        0,                            // Item size
        (destructor)PyDataSetParameters_dealloc,  // Dealloc function
        0,                           // tp_print
        0,                           // tp_getattr
        0,                           // tp_setattr
        0,                           // tp_reserved
        0,                           // tp_repr
        0,                           // tp_as_number
        0,                           // tp_as_sequence
        0,                           // tp_as_mapping
        0,                           // tp_hash  
        0,                           // tp_call
        0,                           // tp_str
        0,                           // tp_getattro
        0,                           // tp_setattro
        0,                           // tp_as_buffer
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, // tp_flags
        "DataSetParameters objects",           // tp_doc
        0,                           // tp_traverse
        0,                           // tp_clear
        0,                           // tp_richcompare
        0,                           // tp_weaklistoffset
        0,                           // tp_iter
        0,                           // tp_iternext
        0,                           // tp_methods
        0,                           // tp_members
        PyDataSetParameters_getsetters,        // tp_getset
        0,                           // tp_base
        0,                           // tp_dict
        0,                           // tp_descr_get
        0,                           // tp_descr_set
        0,                           // tp_dictoffset
        0,                           // tp_init
        0,                           // tp_alloc
        PyDataSetParameters_new,               // tp_new
    };

#pragma endregion

#pragma endregion

typedef struct {
    PyObject_HEAD;
    DataSet * ptrObj;
} PyDataSet;
#pragma region functions
// ---------------------------- Initializers ------------------------------
static int PyDataSet_init(PyDataSet *self, PyObject *args, PyObject *kwds) {
    PyObject *arg, *out = NULL;
    PyArrayObject* arr;
    if(!PyArg_ParseTuple(args,"|O", &out))
    {
        self->ptrObj = new DataSet();
        return 0;
    }
    else
    {
        if(out == NULL)
        {
            
            self->ptrObj = new DataSet();
            return 0; 
        }
        else
        {
            arr = (PyArrayObject*)PyArray_FROM_OTF(out, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
            int ndim =  PyArray_NDIM(arr);
            if(ndim != 2)
            {
                std::cout << "Data object has wrong dimensionality. Expected a 2-D array, recieved " << ndim<< "-D array. DataSet was created, but it is empty." << std::endl;
                Py_DECREF(arr);
                self->ptrObj = new DataSet();
                return 0;
            }
            int no_points = PyArray_DIM(arr,0);
            int size = PyArray_DIM(arr,1);
            if(size > 12)
            {   
                std::cout << "Data object is too large. Expected size of a single point is <= 12. Recieved size :" << size << ". DataSet was created, but it is empty." << std::endl;
                Py_DECREF(arr);
                self->ptrObj = new DataSet();
                return 0;
            }
            double* values = (double*) PyArray_DATA(arr);
            self->ptrObj = new DataSet();
            DataSetParameters parameters = DataSetParameters();
            parameters.NumberOfObjectives = size;
            parameters.nPoints = 0;
            self->ptrObj->setParameters(parameters);
            for(int i = 0; i<no_points; i++)
            {
                Point* p = new Point(size);
                for(int j = 0; j<size; j++)
                {
                    p->ObjectiveValues[j] = values[i*size + j];
                }
                self->ptrObj->add(p);
            }
            Py_DECREF(arr);
            return 0; 

        }
    }
}

static PyObject* PyDataSet_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
        PyDataSet* self = (PyDataSet*)type->tp_alloc(type, 0);
        if (self != NULL) {
            self->ptrObj = new DataSet();  // Create a new C++ object
        }
        return (PyObject*)self;
    }
// --------------------------------- Deallocator --------------------------------------
static void PyDataSet_dealloc(PyPoint * self) {
    delete self->ptrObj;
    Py_TYPE(self)->tp_free(self);
}
// --------------------------------- Type definition ----------------------------------
extern PyTypeObject PyDataSetType;
 // ---------------------------- DataSet methods ---------------------------------------
 static PyObject* PyDataSet_toString(PyObject* self)
 {
     PyDataSet* dataset = (PyDataSet*)self;
     std::string strBuilder = "[";
     for(auto point: dataset->ptrObj->points)
     {
        strBuilder += "[";
        for(int i = 0; i<point->NumberOfObjectives; i++)
        {
            strBuilder += std::to_string(point->ObjectiveValues[i]) + ",";
        }
        strBuilder = strBuilder.substr(0, strBuilder.length()-2) + "],\n";   
     }
     strBuilder = strBuilder.substr(0, strBuilder.length()-2) + "],\n";
     return PyUnicode_FromFormat(strBuilder.c_str());
 }
static PyObject* PyDataSet_add(PyDataSet* self, PyObject* new_point)
{
    try {
        self->ptrObj->add(((PyPoint*)new_point)->ptrObj);
        return Py_True;
    }
    catch (const std::out_of_range&) {
        PyErr_SetString(PyExc_IndexError, "Index out of range");
        return Py_False;
    }
}
static PyDataSetParametersObject* PyDataSet_parameters(PyDataSet* self) {
    PyDataSetParametersObject* params = PyObject_New(PyDataSetParametersObject, &PyDataSetParametersType);
    
    params->ptrObj = self->ptrObj->getParameters();
    return params;
}
static PyObject* PyDataSet_to_numpy(PyDataSet* self, PyObject* new_point)
{
    try {

        std::vector<double> vec;
        for(auto point : self->ptrObj->points)
        for(int i = 0; i< point->NumberOfObjectives; i++)
        {
            vec.push_back( point->ObjectiveValues[i]);
        }


        int length = vec.size();
        npy_intp dims[2];
        dims[0] = self->ptrObj->getParameters()->nPoints;
        dims[1] = self->ptrObj->getParameters()->NumberOfObjectives;
        double* data = new double[length];
        std::copy(vec.begin(), vec.end(), data); 
        PyObject* obj = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, (void*)data);
        return obj;
    }
    catch (const std::out_of_range&) {
        PyErr_SetString(PyExc_IndexError, "Index out of range");
        return Py_False;
    }
}

static PyPoint* PyDataSet_getitem(PyDataSet* self, PyObject* key)
{
    if (!PyLong_Check(key))
    {
        PyErr_SetString(PyExc_TypeError, "Index must be an integer");
        return NULL;
    }

    int index = PyLong_AsLong(key);
    try {
        Point* value = self->ptrObj->get(index);
        PyPoint* point =  PyObject_New(PyPoint, &PyPointType);
        point->ptrObj = value;
        return point;
    }
    catch (const std::out_of_range&) {
        PyErr_SetString(PyExc_IndexError, "Index out of range");
        return NULL;
    }
}


static PyObject* PyDataSet_make_maximalization(PyDataSet* self)
{
    self->ptrObj->typeOfOptimization = DataSet::OptimizationType::maximization;
    return Py_True;
}

static PyObject* PyDataSet_make_minimalization(PyDataSet* self)
{
    self->ptrObj->typeOfOptimization = DataSet::OptimizationType::minimization;
    return Py_True;
}
static int PyDataSet_setitem(PyDataSet* self, PyObject* key, PyObject* value)
{
    if (!PyLong_Check(key))
    {
        PyErr_SetString(PyExc_TypeError, "Index must be an integer and value must be a float.");
        return -1;
    }

    int index = PyLong_AsLong(key);

    if(((PyPoint*)value)->ptrObj->NumberOfObjectives != self->ptrObj->getParameters()->NumberOfObjectives)
    {
        PyErr_SetString(PyExc_TypeError, "Wrong size of the point");
        return -1;
    }
    
    try {
        self->ptrObj->operator[](index) = *((PyPoint*)value)->ptrObj;
        return 0;
    }
    catch (const std::out_of_range&) {
        PyErr_SetString(PyExc_IndexError, "Index out of range");
        return -1;
    }
}
static PyObject* PyDataSet_get_parameters(PyDataSet* self, void* closure) {
    // Assuming parameters are another object type, you'll need to wrap it
    PyDataSetParametersObject* params = PyObject_New(PyDataSetParametersObject, &PyDataSetParametersType);
    if (!params) {
        return PyErr_NoMemory();  // Handle allocation failure
    }
    PyObject_Init((PyObject*)params, &PyDataSetParametersType);  // Ensures proper initialization
    DataSetParameters *dsp = new DataSetParameters();
    dsp->filename = self->ptrObj->getParameters()->filename;
    dsp->name = self->ptrObj->getParameters()->name;
    dsp->NumberOfObjectives = self->ptrObj->getParameters()->NumberOfObjectives;
    dsp->nPoints = self->ptrObj->getParameters()->nPoints;
    dsp->sampleNumber = self->ptrObj->getParameters()->sampleNumber;
    params->ptrObj = dsp;  // Assign valid value
    return (PyObject*)params;
}

static PyObject* PyDataSet_get_ideal(PyDataSet* self, void* closure) {
    // Assuming parameters are another object type, you'll need to wrap it
    PyPoint* point = PyObject_New(PyPoint, &PyPointType);
    if (!point) {
        return PyErr_NoMemory();  // Handle allocation failure
    }
    PyObject_Init((PyObject*)point, &PyPointType);  // Ensures proper initialization
    Point *pointPtr = new Point(*self->ptrObj->getIdeal());
    point->ptrObj = pointPtr;
    return (PyObject*)point;
}

static PyObject* PyDataSet_get_nadir(PyDataSet* self, void* closure) {
    // Assuming parameters are another object type, you'll need to wrap it
    PyPoint* point = PyObject_New(PyPoint, &PyPointType);
    if (!point) {
        return PyErr_NoMemory();  // Handle allocation failure
    }
    PyObject_Init((PyObject*)point, &PyPointType);  // Ensures proper initialization
    Point *pointPtr = new Point(*self->ptrObj->getNadir());
    point->ptrObj = pointPtr;
    return (PyObject*)point;
}

static int PyDataSet_set_parameters(PyDataSet* self, PyObject* value, void* closure) {
    // if (!PyPoint_Check(value)) {  // TODO:Replace with actual Point type check
    //     PyErr_SetString(PyExc_TypeError, "Expected a Point object");
    //     return -1;
    // }

    self->ptrObj->setParameters(*((PyDataSetParametersObject*)value)->ptrObj);  // Replace with actual parameters conversion
    return 0;
}

static int PyDataSet_set_nadir(PyDataSet* self, PyObject* value, void* closure) {
    // if (!PyPoint_Check(value)) {  // TODO:Replace with actual Point type check
    //     PyErr_SetString(PyExc_TypeError, "Expected a Point object");
    //     return -1;
    // }

    self->ptrObj->setNadir(((PyPoint*)value)->ptrObj);  // Replace with actual parameters conversion
    return 0;
}

static int PyDataSet_set_ideal(PyDataSet* self, PyObject* value, void* closure) {
    // if (!PyPoint_Check(value)) {  // TODO:Replace with actual Point type check
    //     PyErr_SetString(PyExc_TypeError, "Expected a Point object");
    //     return -1;
    // }

    self->ptrObj->setIdeal(((PyPoint*)value)->ptrObj);  // Replace with actual parameters conversion
    return 0;
}

static PyGetSetDef PyDataSet_getseters[] = {
    {"nadir", (getter)PyDataSet_get_nadir, (setter)PyDataSet_set_nadir, "nadir", NULL},
    {"ideal", (getter)PyDataSet_get_ideal, (setter)PyDataSet_set_ideal, "ideal", NULL},
    {"parameters", (getter)PyDataSet_get_parameters, (setter)PyDataSet_set_parameters, "parameters", NULL},
    {NULL}
};
static PyMappingMethods PyDataSet_mapping = {
    (lenfunc) NULL,            // No __len__()
    (binaryfunc) PyDataSet_getitem,  // __getitem__()
    (objobjargproc) PyDataSet_setitem,  // __setitem__()
};

static PyMethodDef PyDataSet_methods[] = {
    { "add",(PyCFunction)PyDataSet_add, METH_VARARGS, "add a new point to the dataset" },
    { "to_numpy",(PyCFunction)PyDataSet_to_numpy, METH_VARARGS, "convert to numpy array" },
    { "make_maximalization",(PyCFunction)PyDataSet_make_maximalization, METH_VARARGS, "interpret the dataset as maximalization" },
    { "make_minimalization",(PyCFunction)PyDataSet_make_minimalization, METH_VARARGS, "interpret the dataset as minimalization" },
    { "get_parameters",(PyCFunction)PyDataSet_parameters, METH_VARARGS, "get the parameters object" },

    {NULL}  /* Sentinel */
};

PyTypeObject PyDataSetType = 
 {
    PyVarObject_HEAD_INIT(NULL, 0)
    "moda.DataSet",      // tp_name
    sizeof(PyDataSet), // tp_basicsize
    0,                           // tp_itemsize
    (destructor)PyDataSet_dealloc, // tp_dealloc
    0,                           // tp_print
    0,                           // tp_getattr
    0,                           // tp_setattr
    0,                           // tp_reserved
    0,                           // tp_repr
    0,                           // tp_as_number
    0,                           // tp_as_sequence
    0,                           // tp_as_mapping (newfunc)PyDataSet_mapping
    0,                           // tp_hash
    0,                           // tp_call
    0,                           // tp_str
    0,                           // tp_getattro
    0,                           // tp_setattro
    0,                           // tp_as_buffer
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, // tp_flags
    "DataSet object", // tp_doc
    0,                           // tp_traverse
    0,                           // tp_clear
    0,                           // tp_richcompare
    0,                           // tp_weaklistoffset
    0,                           // tp_iter
    0,                           // tp_iternext
    PyDataSet_methods,                           // tp_methods
    0,                           // tp_members
    PyDataSet_getseters, // tp_getset
    0,  // tp_base
    0,                           // tp_dict
    0,                           // tp_descr_get
    0,                           // tp_descr_set
    0,                           // tp_dictoffset
    (initproc)PyDataSet_init,                           // tp_init
    0,                           // tp_alloc
    PyDataSet_new     // tp_new
};
#pragma endregion
#pragma endregion

#pragma region functions
static PyMethodDef Methods[] = {

    {NULL, NULL, 0, NULL}  // Sentinel
};
#pragma endregion

#pragma region Module Definition
static PyModuleDef modamodule = {
    PyModuleDef_HEAD_INIT,
    "moda",
    "module definition",
    -1,
    Methods, NULL, NULL, NULL, NULL
};
#pragma endregion

#pragma region Package Initializer
// -------------------------- package initializer -------------------------------
PyMODINIT_FUNC PyInit_moda(void)
// create the module
{
    import_array();
    PyObject* moduleObject;
    if (PyType_Ready(&PyPointType) < 0)
        return NULL;
    if (PyType_Ready(&PyDataSetType) < 0)
        return NULL;
    if (PyType_Ready(&PyResultType) < 0)
        return NULL;
    if (PyType_Ready(&PyDataSetParametersType) < 0)
        return NULL;
    moduleObject = PyModule_Create(&modamodule);
    if (moduleObject == NULL)
        return NULL;
    Py_INCREF(&PyResultType);
    Py_INCREF(&PyPointType); 
    PyModule_AddObject(moduleObject, "Result", (PyObject*)&PyResultType);
    PyModule_AddObject(moduleObject, "Point", (PyObject*)&PyPointType);
    PyModule_AddObject(moduleObject, "Dataset", (PyObject*)&PyDataSetType);
    PyModule_AddObject(moduleObject, "DatasetParameters", (PyObject*)&PyDataSetParametersType);

    return moduleObject;
}
#pragma endregion

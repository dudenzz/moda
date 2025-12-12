//Algorithms and Data Structures for Multi-objective Optimization

#include <Python.h>
#include "numpy/arrayobject.h"
#include <cstdio>
#include "Solver.h"
#include "QR2.h"
#include "HSSSolver.h"
#include "DBHVESolver.h"
#include "MCHVESolver.h"
#include "QEHCSolver.h"
#include "QHV_BQ.h"
#include "IQHVSolver.h"
#include "DataSet.h"
#include "Point.h"
#include "DataSet.h"
#include "Result.h"
#include "Helpers.h"
#include "DataSetParameters.h"
#include "SolverParameters.h"
#include "iostream"

// using qhv::IQHV;
using namespace moda;


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
#pragma region data structures
typedef struct {
    PyObject_HEAD
    Solver * ptrObj;
    std::string type;
} PySolver;

typedef struct {
    PyObject_HEAD;
    Point * ptrObj;
} PyPoint;

typedef struct {
    PyObject_HEAD;
    DataSet * ptrObj;
} PyDataSet;
typedef struct {
    PyObject_HEAD
    DataSetParameters* ptrObj;  // Pointer to the actual C++ DataSetParameters object
} PyDataSetParametersObject;

typedef struct {
    PyObject_HEAD
    SwitchSettings* settings;  // C++ SwitchSettings object
} PySwitchSettingsObject;

typedef struct {
    PyObject_HEAD
    SolverSettings* settings;  // C++ SolverSettings object
} PySolverSettingsObject;

typedef struct {
    PyResultObject base;
    SubsetSelectionResult* result;// Pointer to C++ SubsetSelectionResult
} PySubsetSelectionResultObject;
#pragma endregion
#pragma region Point
// ----------------------- Initializer ----------------------------------
static int PyPoint_init(PyPoint *self, PyObject *args, PyObject *kwds)
{
    PyObject *arg, *arr, *out = NULL;

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
            arr = PyArray_FROM_OTF(out, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
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
// --------------------------------- Deallocator --------------------------------------
static void PyPoint_dealloc(PyPoint * self)
{
    delete self->ptrObj;
    Py_TYPE(self)->tp_free(self);
}
// --------------------------------- Type definition ----------------------------------
static PyTypeObject PyPointType = {
    PyVarObject_HEAD_INIT(NULL, 0)
     "moda.Point"   /* tp_name */,
 };
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
    
    // Parse the arguments to get the other point
    if (!PyArg_ParseTuple(args, "O", &otherPointObj))
        return NULL;

    // Ensure the other object is actually a PyPoint object
    if (!PyObject_TypeCheck(otherPointObj, &PyPointType))
    {
        PyErr_SetString(PyExc_TypeError, "Argument must be a Point object");
        return NULL;
    }

    // Call the C++ Compare method
    ComparisonResult result = self->ptrObj->Compare(*otherPointObj->ptrObj);

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

static PyMappingMethods PyPoint_mapping = {
    (lenfunc) NULL,            // No __len__()
    (binaryfunc) PyPoint_getitem,  // __getitem__()
    (objobjargproc) PyPoint_setitem,  // __setitem__()
};

#pragma endregion
#pragma region SwitchSettings
    extern PyTypeObject PySwitchSettingsType;
    extern PyTypeObject PyDataSetType;
    // Destructor for PySwitchSettings
    static void PySwitchSettings_dealloc(PySwitchSettingsObject* self) {
        delete self->settings;  // Clean up the C++ object when the Python object is deleted
        Py_TYPE(self)->tp_free((PyObject*)self);
    }

    // Constructor for PySwitchSettings
    static PyObject* PySwitchSettings_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
        PySwitchSettingsObject* self = (PySwitchSettingsObject*)type->tp_alloc(type, 0);
        if (self != NULL) {
            self->settings = new SwitchSettings();  // Create a new C++ object
        }
        return (PyObject*)self;
    }

    // Initialize PySwitchSettings
    static int PySwitchSettings_init(PySwitchSettingsObject* self, PyObject* args, PyObject* kwds) {
        static const char* kwlist[] = {"switchTime", "gap", "maxStackProblemSize", "iterations", nullptr};
        if (!PyArg_ParseTupleAndKeywords(args, kwds, "|idii", (char**)kwlist,
                                          &self->settings->switchTime,
                                          &self->settings->gap,
                                          &self->settings->maxStackProblemSize,
                                          &self->settings->iterations)) {
            return -1;
        }
        return 0;
    }

    // Getter functions for SwitchSettings attributes
    static PyObject* PySwitchSettings_get_switchTime(PySwitchSettingsObject* self, void* closure) {
        return PyLong_FromLong(self->settings->switchTime);
    }

    static PyObject* PySwitchSettings_get_gap(PySwitchSettingsObject* self, void* closure) {
        return PyFloat_FromDouble(self->settings->gap);
    }

    static PyObject* PySwitchSettings_get_maxStackProblemSize(PySwitchSettingsObject* self, void* closure) {
        return PyLong_FromLong(self->settings->maxStackProblemSize);
    }

    static PyObject* PySwitchSettings_get_iterations(PySwitchSettingsObject* self, void* closure) {
        return PyLong_FromLong(self->settings->iterations);
    }


    // Method to get default settings
    static PyObject* PySwitchSettings_defaultSettings(PyObject* self, PyObject* args) {
        SwitchSettings dSett = SwitchSettings::defaultSettings();
        PySwitchSettingsObject* pySett = (PySwitchSettingsObject*)PySwitchSettings_new(&PySwitchSettingsType, NULL, NULL);
        pySett->settings = new SwitchSettings(dSett);
        Py_INCREF(pySett);
        return (PyObject*)pySett;
    }
    // Define the getter and setter methods for SwitchSettings
    static PyGetSetDef PySwitchSettings_getsetters[] = {
        {"switchTime", (getter)PySwitchSettings_get_switchTime, NULL, "Switch time", NULL},
        {"gap", (getter)PySwitchSettings_get_gap, NULL, "Gap", NULL},
        {"maxStackProblemSize", (getter)PySwitchSettings_get_maxStackProblemSize, NULL, "Max Stack Problem Size", NULL},
        {"iterations", (getter)PySwitchSettings_get_iterations, NULL, "Iterations", NULL},
        {NULL}  // Sentinel
    };

    // Methods of the SwitchSettings class
    static PyMethodDef PySwitchSettings_methods[] = {
        {"defaultSettings", (PyCFunction)PySwitchSettings_defaultSettings, METH_NOARGS, "Returns default settings."},
        {NULL}  // Sentinel
    };

    // Type definition for PySwitchSettings
    static PyTypeObject PySwitchSettingsType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "moda.SwitchSettings",  // Name of the Python class
        sizeof(PySwitchSettingsObject),
        0,  // tp_itemsize
        (destructor)PySwitchSettings_dealloc,  // tp_dealloc
        0,  // tp_print
        0,  // tp_getattr
        0,  // tp_setattr
        0,  // tp_reserved
        0,  // tp_repr
        0,  // tp_as_number
        0,  // tp_as_sequence
        0,  // tp_as_mapping
        0,  // tp_hash
        0,  // tp_call
        0,  // tp_str
        0,  // tp_getattro
        0,  // tp_setattro
        0,  // tp_as_buffer
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,  // tp_flags
        "SwitchSettings objects",  // tp_doc
        0,  // tp_traverse
        0,  // tp_clear
        0,  // tp_richcompare
        0,  // tp_weaklistoffset
        0,  // tp_iter
        0,  // tp_iternext
        PySwitchSettings_methods,  // tp_methods
        0,  // tp_members
        PySwitchSettings_getsetters,  // tp_getset
        0,  // tp_base
        0,  // tp_dict
        0,  // tp_descr_get
        0,  // tp_descr_set
        0,  // tp_dictoffset
        (initproc)PySwitchSettings_init,  // tp_init
        0,  // tp_alloc
        PySwitchSettings_new,  // tp_new
    };


#pragma endregion
#pragma region SolverSettings


extern PyTypeObject PySolverSettingsType;

// Destructor for PySolverSettings
static void PySolverSettings_dealloc(PySolverSettingsObject* self) {
    delete self->settings;  // Clean up the C++ object when the Python object is deleted
    Py_TYPE(self)->tp_free((PyObject*)self);
}

// Constructor for PySolverSettings
static PyObject* PySolverSettings_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    PySolverSettingsObject* self = (PySolverSettingsObject*)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->settings = new SolverSettings();  // Create a new C++ object
    }
    return (PyObject*)self;
}

// Initialize PySolverSettings
static int PySolverSettings_init(PySolverSettingsObject* self, PyObject* args, PyObject* kwds) {
    static const char* kwlist[] = {"dataset","callbacks", "worseReferencePoint", "betterReferencePoint", "MonteCarlo", "minimize",
                                   "sort", "calculateHV", "shuffle", "offset", "iterationsLimit", "mcSettings", 
                                   "MaxEstimationTime", "SaveInterval", "StoppingCriteria", "StoppingSubsetSize", 
                                   "StoppingTime", "seed", "useOriginalNDTree", "experimental", "exhaustive", "incremental", nullptr};
    PyDataSet* dataset_object;
    PyPoint* worseReferencePoint = NULL, *betterReferencePoint = NULL;
    PySwitchSettingsObject* switchSettings = NULL;

    bool callbacks = true, monteCarlo = false, minimize = true, sort = true, calculateHV = true,useOriginalNDTree = false, shuffle = true,experimental = false,exhaustive=false,subsetSelectionQHVIncremental = false;
    int offset = 0, iterationsLimit = 10000,maxEstimationTime = 100000, saveInterval = 1000, stoppingCriteria = 1, stoppingSubsetSize = 5, stoppingTime = 10000, seed = 0, callbackIterations = 1000;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|$pOOppppbbObbbbbbbpbbbb", (char**)kwlist,
                                      &dataset_object, 
                                      callbacks,
                                      &worseReferencePoint,
                                      &betterReferencePoint,
                                      monteCarlo,
                                      minimize,
                                      sort,
                                      calculateHV,
                                      shuffle,
                                      offset,
                                      iterationsLimit,
                                      &switchSettings,
                                      maxEstimationTime,
                                      saveInterval,
                                      stoppingCriteria,
                                      stoppingSubsetSize,
                                      stoppingTime,
                                      seed,
                                      useOriginalNDTree,
                                      experimental,
                                      exhaustive,
                                      subsetSelectionQHVIncremental,
                                      callbackIterations)) {
        return -1;
    }
    

    // Ensure obj is of type PyDataSet
    if (!PyObject_TypeCheck(dataset_object, &PyDataSetType)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be a DataSet object");
        return NULL;
    }
    if (dataset_object->ptrObj == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "DataSet object is uninitialized");
        return NULL;
    }

    // self->settings = new SolverSettings(*dataset_object->ptrObj, SolverSettings::zeroone);
    self->settings = new SolverSettings();

    //objects
    if(worseReferencePoint != NULL) self->settings->worseReferencePoint = worseReferencePoint->ptrObj;
    if(betterReferencePoint != NULL) self->settings->betterReferencePoint =  betterReferencePoint->ptrObj;
    if(switchSettings != NULL) self->settings->SwitchToMCSettings =  *switchSettings->settings;

 
    //booleans
    self->settings->callbacks = callbacks;
    self->settings->MonteCarlo = monteCarlo;
    self->settings->minimize = minimize;
    self->settings->sort = sort;
    self->settings->calculateHV = calculateHV;
    self->settings->useOriginalNDTree = useOriginalNDTree;
    self->settings->shuffle = shuffle;
    self->settings->experimental = experimental;
    self->settings->exhaustive = exhaustive;
    self->settings->SubsetSelectionQHVIncremental = subsetSelectionQHVIncremental;


    //integers
    self->settings->offset = offset;
    self->settings->iterationsLimit = iterationsLimit;
    self->settings->MaxEstimationTime = maxEstimationTime;
    self->settings->SaveInterval = saveInterval;
    self->settings->StoppingCriteria = stoppingCriteria;
    self->settings->StoppingSubsetSize = stoppingSubsetSize;
    self->settings->StoppingTime = stoppingTime;
    self->settings->seed = seed;
    self->settings->callbackIterations = callbackIterations;

    return 0;
}



// Method to create default SolverSettings object
static PyObject* PySolverSettings_defaultSettings(PyObject* self, PyObject* args) {
    SolverSettings dSett;
    // Call the constructor to populate the default settings
    PySolverSettingsObject* pySett = (PySolverSettingsObject*)PySolverSettings_new(&PySolverSettingsType, NULL, NULL);
    pySett->settings = new SolverSettings(dSett);
    Py_INCREF(pySett);
    return (PyObject*)pySett;
}
// getters and setters

static PyObject* PySolverSettings_get_callbacks(PySolverSettingsObject* self, void* closure) {
    return PyBool_FromLong(self->settings->callbacks);
}

static int PySolverSettings_set_callbacks(PySolverSettingsObject* self, PyObject* value, void* closure) {
    if (value == NULL || !PyBool_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Expected a boolean value");
        return -1;
    }
    self->settings->callbacks = (value == Py_True);
    return 0;
}
static PyObject* PySolverSettings_get_SubsetSelectionQHVIncremental(PySolverSettingsObject* self, void* closure) {
    return PyBool_FromLong(self->settings->SubsetSelectionQHVIncremental);
}

static int PySolverSettings_set_SubsetSelectionQHVIncremental(PySolverSettingsObject* self, PyObject* value, void* closure) {
    if (value == NULL || !PyBool_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Expected a boolean value");
        return -1;
    }
    self->settings->SubsetSelectionQHVIncremental = (value == Py_True);
    return 0;
}
static PyObject* PySolverSettings_get_worseReferencePoint(PySolverSettingsObject* self, void* closure) {
    // Assuming Point is another object type, you'll need to wrap it
    PyPoint* point = PyObject_New(PyPoint, &PyPointType);
    if (!point) {
        return PyErr_NoMemory();  // Handle allocation failure
    }
    PyObject_Init((PyObject*)point, &PyPointType);  // Ensures proper initialization
    point->ptrObj = new Point(*self->settings->worseReferencePoint);  // Assign valid value
    return (PyObject*)point;
}

static int PySolverSettings_set_worseReferencePoint(PySolverSettingsObject* self, PyObject* value, void* closure) {
    // if (!PyPoint_Check(value)) {  // TODO:Replace with actual Point type check
    //     PyErr_SetString(PyExc_TypeError, "Expected a Point object");
    //     return -1;
    // }

    self->settings->worseReferencePoint = ((PyPoint*)value)->ptrObj;  // Replace with actual Point conversion
    return 0;
}

static PyObject* PySolverSettings_get_betterReferencePoint(PySolverSettingsObject* self, void* closure) {
    // Assuming Point is another object type, you'll need to wrap it
    PyPoint* point = PyObject_New(PyPoint, &PyPointType);
    if (!point) {
        return PyErr_NoMemory();  // Handle allocation failure
    }
    PyObject_Init((PyObject*)point, &PyPointType);  // Ensures proper initialization
    point->ptrObj = new Point(*self->settings->betterReferencePoint);  // Assign valid value
    return (PyObject*)point;
}

static int PySolverSettings_set_betterReferencePoint(PySolverSettingsObject* self, PyObject* value, void* closure) {
    // if (!PyPoint_Check(value)) {  // TODO: Replace with actual Point type check
    //     PyErr_SetString(PyExc_TypeError, "Expected a Point object");
    //     return -1;
    // }
    self->settings->betterReferencePoint = ((PyPoint*)value)->ptrObj;  // Replace with actual Point conversion
    return 0;
}

static PyObject* PySolverSettings_get_MonteCarlo(PySolverSettingsObject* self, void* closure) {
    return PyBool_FromLong(self->settings->MonteCarlo);
}

static int PySolverSettings_set_MonteCarlo(PySolverSettingsObject* self, PyObject* value, void* closure) {
    if (value == NULL || !PyBool_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Expected a boolean value");
        return -1;
    }
    self->settings->MonteCarlo = (value == Py_True);
    return 0;
}

static PyObject* PySolverSettings_get_minimize(PySolverSettingsObject* self, void* closure) {
    return PyBool_FromLong(self->settings->minimize);
}

static int PySolverSettings_set_minimize(PySolverSettingsObject* self, PyObject* value, void* closure) {
    if (value == NULL || !PyBool_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Expected a boolean value");
        return -1;
    }
    self->settings->minimize = (value == Py_True);
    return 0;
}

static PyObject* PySolverSettings_get_sort(PySolverSettingsObject* self, void* closure) {
    return PyBool_FromLong(self->settings->sort);
}

static int PySolverSettings_set_sort(PySolverSettingsObject* self, PyObject* value, void* closure) {
    if (value == NULL || !PyBool_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Expected a boolean value");
        return -1;
    }
    self->settings->sort = (value == Py_True);
    return 0;
}

static PyObject* PySolverSettings_get_calculateHV(PySolverSettingsObject* self, void* closure) {
    return PyBool_FromLong(self->settings->calculateHV);
}

static int PySolverSettings_set_calculateHV(PySolverSettingsObject* self, PyObject* value, void* closure) {
    if (value == NULL || !PyBool_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Expected a boolean value");
        return -1;
    }
    self->settings->calculateHV = (value == Py_True);
    return 0;
}

static PyObject* PySolverSettings_get_shuffle(PySolverSettingsObject* self, void* closure) {
    return PyBool_FromLong(self->settings->shuffle);
}

static int PySolverSettings_set_shuffle(PySolverSettingsObject* self, PyObject* value, void* closure) {
    if (value == NULL || !PyBool_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Expected a boolean value");
        return -1;
    }
    self->settings->shuffle = (value == Py_True);
    return 0;
}

static PyObject* PySolverSettings_get_offset(PySolverSettingsObject* self, void* closure) {
    return PyLong_FromLong(self->settings->offset);
}

static int PySolverSettings_set_offset(PySolverSettingsObject* self, PyObject* value, void* closure) {
    if (value == NULL || !PyLong_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Expected an integer value");
        return -1;
    }
    self->settings->offset = (int)PyLong_AsLong(value);
    return 0;
}

static PyObject* PySolverSettings_get_iterationsLimit(PySolverSettingsObject* self, void* closure) {
    return PyLong_FromUnsignedLong(self->settings->iterationsLimit);
}

static int PySolverSettings_set_iterationsLimit(PySolverSettingsObject* self, PyObject* value, void* closure) {
    if (value == NULL || !PyLong_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Expected an integer value");
        return -1;
    }
    self->settings->iterationsLimit = (unsigned long)PyLong_AsUnsignedLong(value);
    return 0;
}

static PyObject* PySolverSettings_get_MaxEstimationTime(PySolverSettingsObject* self, void* closure) {
    return PyLong_FromLong(self->settings->MaxEstimationTime);
}

static int PySolverSettings_set_MaxEstimationTime(PySolverSettingsObject* self, PyObject* value, void* closure) {
    if (value == NULL || !PyLong_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Expected an integer value");
        return -1;
    }
    self->settings->MaxEstimationTime = (int)PyLong_AsLong(value);
    return 0;
}

static PyObject* PySolverSettings_get_SaveInterval(PySolverSettingsObject* self, void* closure) {
    return PyLong_FromLong(self->settings->SaveInterval);
}

static int PySolverSettings_set_SaveInterval(PySolverSettingsObject* self, PyObject* value, void* closure) {
    if (value == NULL || !PyLong_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Expected an integer value");
        return -1;
    }
    self->settings->SaveInterval = (int)PyLong_AsLong(value);
    return 0;
}


static PyObject* PySolverSettings_get_StoppingSubsetSize(PySolverSettingsObject* self, void* closure) {
    return PyLong_FromLong(self->settings->StoppingSubsetSize);
}

static int PySolverSettings_set_StoppingSubsetSize(PySolverSettingsObject* self, PyObject* value, void* closure) {
    if (value == NULL || !PyLong_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Expected an integer value");
        return -1;
    }
    self->settings->StoppingSubsetSize = (int)PyLong_AsLong(value);
    return 0;
}
static PyGetSetDef PySolverSettings_getseters[] = {
    {"callbacks", (getter)PySolverSettings_get_callbacks, (setter)PySolverSettings_set_callbacks, "callbacks", NULL},
    {"SubsetSelectionQHVIncremental", (getter)PySolverSettings_get_SubsetSelectionQHVIncremental, (setter)PySolverSettings_set_SubsetSelectionQHVIncremental, "SubsetSelectionQHVIncremental", NULL},
    {"worseReferencePoint", (getter)PySolverSettings_get_worseReferencePoint, (setter)PySolverSettings_set_worseReferencePoint, "worseReferencePoint", NULL},
    {"betterReferencePoint", (getter)PySolverSettings_get_betterReferencePoint, (setter)PySolverSettings_set_betterReferencePoint, "betterReferencePoint", NULL},
    {"MonteCarlo", (getter)PySolverSettings_get_MonteCarlo, (setter)PySolverSettings_set_MonteCarlo, "MonteCarlo", NULL},
    {"minimize", (getter)PySolverSettings_get_minimize, (setter)PySolverSettings_set_minimize, "minimize", NULL},
    {"sort", (getter)PySolverSettings_get_sort, (setter)PySolverSettings_set_sort, "sort", NULL},
    {"calculateHV", (getter)PySolverSettings_get_calculateHV, (setter)PySolverSettings_set_calculateHV, "calculateHV", NULL},
    {"shuffle", (getter)PySolverSettings_get_shuffle, (setter)PySolverSettings_set_shuffle, "shuffle", NULL},
    {"offset", (getter)PySolverSettings_get_offset, (setter)PySolverSettings_set_offset, "offset", NULL},
    {"iterationsLimit", (getter)PySolverSettings_get_iterationsLimit, (setter)PySolverSettings_set_iterationsLimit, "iterationsLimit", NULL},
    {"MaxEstimationTime", (getter)PySolverSettings_get_MaxEstimationTime, (setter)PySolverSettings_set_MaxEstimationTime, "MaxEstimationTime", NULL},
    {"SaveInterval", (getter)PySolverSettings_get_SaveInterval, (setter)PySolverSettings_set_SaveInterval, "SaveInterval", NULL},
    {"StoppingSubsetSize", (getter)PySolverSettings_get_StoppingSubsetSize, (setter)PySolverSettings_set_StoppingSubsetSize, "StoppingSubsetSize", NULL},
    {NULL}  // Sentinel
};
// Define the PySolverSettingsType
static PyTypeObject PySolverSettingsType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "moda.SolverSettings",  // Name of the Python class
    sizeof(PySolverSettingsObject),
    0,  // tp_itemsize
    (destructor)PySolverSettings_dealloc,  // tp_dealloc
    0,  // tp_print
    0,  // tp_getattr
    0,  // tp_setattr
    0,  // tp_reserved
    0,  // tp_repr
    0,  // tp_as_number
    0,  // tp_as_sequence
    0,  // tp_as_mapping
    0,  // tp_hash
    0,  // tp_call
    0,  // tp_str
    0,  // tp_getattro
    0,  // tp_setattro
    0,  // tp_as_buffer
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,  // tp_flags
    "SolverSettings objects",  // tp_doc
    0,  // tp_traverse
    0,  // tp_clear
    0,  // tp_richcompare
    0,  // tp_weaklistoffset
    0,  // tp_iter
    0,  // tp_iternext
    0,  // tp_methods
    0,  // tp_members
    PySolverSettings_getseters,  // tp_getset
    0,  // tp_base
    0,  // tp_dict
    0,  // tp_descr_get
    0,  // tp_descr_set
    0,  // tp_dictoffset
    (initproc)PySolverSettings_init,  // tp_init
    0,  // tp_alloc
    PySolverSettings_new,  // tp_new
};
#pragma endregion
#pragma region DataSetParameters
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

    static PyObject* PyDataSetParameters_get_normalize(PyDataSetParametersObject* self, void* closure) {
        return PyBool_FromLong(self->ptrObj->normalize);
    }

    static int PyDataSetParameters_set_normalize(PyDataSetParametersObject* self, PyObject* value, void* closure) {
        if (!self || !self->ptrObj) {
            PyErr_SetString(PyExc_AttributeError, "PyDataSetParametersObject or its ptrObj is NULL");
            return -1;
        }
    
        // Ensure the value is a boolean
        if (!PyBool_Check(value)) {
            PyErr_SetString(PyExc_TypeError, "The attribute 'normalize' must be a boolean");
            return -1;
        }
    
        // Set the normalize field
        self->ptrObj->normalize = PyObject_IsTrue(value);
    
        return 0;  // Success
    }

    static PyGetSetDef PyDataSetParameters_getsetters[] = {
        {"filename", (getter)PyDataSetParameters_get_filename, NULL, "Filename", NULL},
        {"name", (getter)PyDataSetParameters_get_name, NULL, "Experiment name", NULL},
        {"NumberOfObjectives", (getter)PyDataSetParameters_get_NumberOfObjectives, NULL, "Number of objectives", NULL},
        {"nPoints", (getter)PyDataSetParameters_get_nPoints, NULL, "Number of points", NULL},
        {"sampleNumber", (getter)PyDataSetParameters_get_sampleNumber, NULL, "Sample number", NULL},
        {"normalize", (getter)PyDataSetParameters_get_normalize, (setter)PyDataSetParameters_set_normalize, "Normalize dataset", NULL},
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
#pragma region Result

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

// Define properties
static PyGetSetDef PyResult_getsetters[] = {
    {"elapsed_time", (getter)PyResult_get_elapsed_time, (setter)PyResult_set_elapsed_time, "Elapsed Time", NULL},
    {"final_result", (getter)PyResult_get_final_result, (setter)PyResult_set_final_result, "Final Result", NULL},
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
#pragma region HypervolumeResult
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
    return PyFloat_FromDouble(self->result->Volume);
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
#pragma region SubsetSelectionResult
// Constructor for SubsetSelectionResult
static PyObject* PySubsetSelectionResult_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    PySubsetSelectionResultObject* self;
    self = (PySubsetSelectionResultObject*)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->result = new SubsetSelectionResult();
        self->base.result = self->result;  // Point base class to the subclass object
    }
    return (PyObject*)self;
}

// Getter for volume in SubsetSelection
static PyObject* PySubsetSelectionResult_get_volume(PySubsetSelectionResultObject* self, void* closure) {
    return PyFloat_FromDouble(self->result->Volume);
}
static PyObject* PySubsetSelectionResult_get_selectedPoints(PySubsetSelectionResultObject* self, PyObject* args) {
    // Convert std::vector<int> to Python list
    PyObject* pyList = PyList_New(self->result->selectedPoints.size());
    for (size_t i = 0; i < self->result->selectedPoints.size(); ++i) {
        PyList_SetItem(pyList, i, PyLong_FromLong(self->result->selectedPoints[i]));
    }
    return pyList;
}
// Define getters and setters for SubsetSelection
static PyGetSetDef PySubsetSelectionResult_getsetters[] = {
    {"volume", (getter)PySubsetSelectionResult_get_volume, NULL, "Hypervolume", NULL},
    {"subset", (getter)PySubsetSelectionResult_get_selectedPoints, NULL, "Subset", NULL},
    {NULL}  // Sentinel
};

// Define the type for SubsetSelection
static PyTypeObject PySubsetSelectionResultType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "moda.SubsetSelectionResult",      // tp_name
    sizeof(PySubsetSelectionResultObject), // tp_basicsize
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
    "SubsetSelectionResult objects", // tp_doc
    0,                           // tp_traverse
    0,                           // tp_clear
    0,                           // tp_richcompare
    0,                           // tp_weaklistoffset
    0,                           // tp_iter
    0,                           // tp_iternext
    0,                           // tp_methods
    0,                           // tp_members
    PySubsetSelectionResult_getsetters, // tp_getset
    (PyTypeObject*)&PyResultType,  // tp_base
    0,                           // tp_dict
    0,                           // tp_descr_get
    0,                           // tp_descr_set
    0,                           // tp_dictoffset
    0,                           // tp_init
    0,                           // tp_alloc
    PySubsetSelectionResult_new,     // tp_new
};



// Module methods
static PyMethodDef SubsetSelectionResult_methods[] = {
    {NULL, NULL, 0, NULL}  // Sentinel
};
#pragma endregion
#pragma region ContributionResult
// Constructor for ContributionResult
static PyObject* PyContributionResult_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    PyContributionResultObject* self;
    self = (PyContributionResultObject*)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->result = new ContributionResult();
        self->base.result = self->result;  // Point base class to the subclass object
    }
    return (PyObject*)self;
}

// Getter for volume in ContributionResult
static PyObject* PyContributionResult_get_maxctr(PyContributionResultObject* self, void* closure) {
    return PyFloat_FromDouble(self->result->MaximumContribution);
}
static PyObject* PyContributionResult_get_minctr(PyContributionResultObject* self, void* closure) {
    return PyFloat_FromDouble(self->result->MinimumContribution);
}
static PyObject* PyContributionResult_get_maxctr_idx(PyContributionResultObject* self, void* closure) {
    return PyLong_FromLong(self->result->MaximumContributionIndex);
}
static PyObject* PyContributionResult_get_minctr_idx(PyContributionResultObject* self, void* closure) {
    return PyLong_FromLong(self->result->MinimumContributionIndex);
}

// Define getters and setters for ContributionResult
static PyGetSetDef PyContributionResult_getsetters[] = {
    {"maxContribution", (getter)PyContributionResult_get_maxctr, NULL, "maxContribution", NULL},
    {"minContribution", (getter)PyContributionResult_get_minctr, NULL, "minContribution", NULL},
    {"maxContributionIndex", (getter)PyContributionResult_get_maxctr_idx, NULL, "maxContributionIndex", NULL},
    {"minContributionIndex", (getter)PyContributionResult_get_minctr_idx, NULL, "minContributionIndex", NULL},
    {NULL}  // Sentinel
};

// Define the type for ContributionResult
static PyTypeObject PyContributionResultType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "moda.ContributionResult",      // tp_name
    sizeof(PyContributionResultObject), // tp_basicsize
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
    "ContributionResult objects", // tp_doc
    0,                           // tp_traverse
    0,                           // tp_clear
    0,                           // tp_richcompare
    0,                           // tp_weaklistoffset
    0,                           // tp_iter
    0,                           // tp_iternext
    0,                           // tp_methods
    0,                           // tp_members
    PyContributionResult_getsetters, // tp_getset
    (PyTypeObject*)&PyResultType,  // tp_base
    0,                           // tp_dict
    0,                           // tp_descr_get
    0,                           // tp_descr_set
    0,                           // tp_dictoffset
    0,                           // tp_init
    0,                           // tp_alloc
    PyContributionResult_new,     // tp_new
};



// Module methods
static PyMethodDef ContributionResult_methods[] = {
    {NULL, NULL, 0, NULL}  // Sentinel
};
#pragma endregion
#pragma region BoundedResult
// Constructor for BoundedResult
static PyObject* PyBoundedResult_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    PyBoundedResultObject* self;
    self = (PyBoundedResultObject*)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->result = new BoundedResult();
        self->base.result = self->result;  // Point base class to the subclass object
    }
    return (PyObject*)self;
}

// Getter for volume in BoundedResult
static PyObject* PyBoundedResult_get_volume(PyBoundedResultObject* self, void* closure) {
    return PyFloat_FromDouble(self->result->VolumeEstimation);
}
static PyObject* PyBoundedResult_get_lower(PyBoundedResultObject* self, void* closure) {
    return PyFloat_FromDouble(self->result->LowerBound);
}
static PyObject* PyBoundedResult_get_upper(PyBoundedResultObject* self, void* closure) {
    return PyFloat_FromDouble(self->result->UpperBound);
}
static PyObject* PyBoundedResult_get_grnt(PyBoundedResultObject* self, void* closure) {
    return Py_BuildValue("p",self->result->Guaranteed);
}
// Define getters and setters for BoundedResult
static PyGetSetDef PyBoundedResult_getsetters[] = {
    {"volume_estimation", (getter)PyBoundedResult_get_volume, NULL, "volume_estimation", NULL},
    {"lower_bound", (getter)PyBoundedResult_get_lower, NULL, "lower_bound", NULL},
    {"upper_bound", (getter)PyBoundedResult_get_volume, NULL, "upper_bound", NULL},
    {"guaranteed", (getter)PyBoundedResult_get_grnt, NULL, "guaranteed", NULL},
    {NULL}  // Sentinel
};


// Define the type for BoundedResult
static PyTypeObject PyBoundedResultType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "moda.BoundedResult",      // tp_name
    sizeof(PyBoundedResultObject), // tp_basicsize
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
    "BoundedResult objects", // tp_doc
    0,                           // tp_traverse
    0,                           // tp_clear
    0,                           // tp_richcompare
    0,                           // tp_weaklistoffset
    0,                           // tp_iter
    0,                           // tp_iternext
    0,                           // tp_methods
    0,                           // tp_members
    PyBoundedResult_getsetters, // tp_getset
    (PyTypeObject*)&PyResultType,  // tp_base
    0,                           // tp_dict
    0,                           // tp_descr_get
    0,                           // tp_descr_set
    0,                           // tp_dictoffset
    0,                           // tp_init
    0,                           // tp_alloc
    PyBoundedResult_new,     // tp_new
};



// Module methods
static PyMethodDef BoundedResult_methods[] = {
    {NULL, NULL, 0, NULL}  // Sentinel
};
#pragma endregion
#pragma region R2Result
// Constructor for R2Result
static PyObject* PyR2Result_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    PyR2ResultObject* self;
    self = (PyR2ResultObject*)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->result = new R2Result();
        self->base.result = self->result;  // Point base class to the subclass object
    }
    return (PyObject*)self;
}

// Getter for volume in R2Result
static PyObject* PyR2Result_get_volume(PyR2ResultObject* self, void* closure) {
    return PyFloat_FromDouble(self->result->Hypervolume);
}
static PyObject* PyR2Result_get_r2(PyR2ResultObject* self, void* closure) {
    return PyFloat_FromDouble(self->result->R2);
}
// Define getters and setters for R2Result
static PyGetSetDef PyR2Result_getsetters[] = {
    {"R2", (getter)PyR2Result_get_r2, NULL, "R2", NULL},
    {"volume", (getter)PyR2Result_get_volume, NULL, "Hypervolume", NULL},
    {NULL}  // Sentinel
};

// Define the type for R2Result
static PyTypeObject PyR2ResultType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "moda.R2Result",      // tp_name
    sizeof(PyR2ResultObject), // tp_basicsize
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
    "R2Result objects", // tp_doc
    0,                           // tp_traverse
    0,                           // tp_clear
    0,                           // tp_richcompare
    0,                           // tp_weaklistoffset
    0,                           // tp_iter
    0,                           // tp_iternext
    0,                           // tp_methods
    0,                           // tp_members
    PyR2Result_getsetters, // tp_getset
    (PyTypeObject*)&PyResultType,  // tp_base
    0,                           // tp_dict
    0,                           // tp_descr_get
    0,                           // tp_descr_set
    0,                           // tp_dictoffset
    0,                           // tp_init
    0,                           // tp_alloc
    PyR2Result_new,     // tp_new
};
#pragma endregion
#pragma region DataSet
// ---------------------------- Initializers ------------------------------
static int PyDataSet_init(PyDataSet *self, PyObject *args, PyObject *kwds)
{
    PyObject *arg, *arr, *out = NULL;

    if(!PyArg_ParseTuple(args,"|O", &out))
    {
        self->ptrObj = new DataSet(3);
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
            arr = PyArray_FROM_OTF(out, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
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
// --------------------------------- Deallocator --------------------------------------
static void PyDataSet_dealloc(PyPoint * self)
{
    delete self->ptrObj;
    Py_TYPE(self)->tp_free(self);
}
// --------------------------------- Type definition ----------------------------------
static PyTypeObject PyDataSetType = {
    PyVarObject_HEAD_INIT(NULL, 0)
     "moda.DataSet"   /* tp_name */,
 };
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
    
    params->ptrObj = &self->ptrObj->getParameters();
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
        dims[0] = self->ptrObj->getParameters().nPoints;
        dims[1] = self->ptrObj->getParameters().NumberOfObjectives;
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
static PyObject* PyDataSet_disable_normalization(PyDataSet* self)
{
    self->ptrObj->setNormalize(false);
    return Py_True;
}

static PyObject* PyDataSet_make_maximalization(PyDataSet* self)
{
    self->ptrObj->typeOfOptimization = DataSet::OptimizationType::maximalization;
    return Py_True;
}

static PyObject* PyDataSet_make_minimalization(PyDataSet* self)
{
    self->ptrObj->typeOfOptimization = DataSet::OptimizationType::minimalization;
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

    if(((PyPoint*)value)->ptrObj->NumberOfObjectives != self->ptrObj->getParameters().NumberOfObjectives)
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
    dsp->filename = self->ptrObj->getParameters().filename;
    dsp->name = self->ptrObj->getParameters().name;
    dsp->NumberOfObjectives = self->ptrObj->getParameters().NumberOfObjectives;
    dsp->nPoints = self->ptrObj->getParameters().nPoints;
    dsp->sampleNumber = self->ptrObj->getParameters().sampleNumber;
    dsp->normalize = self->ptrObj->getParameters().normalize;
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
    { "disable_normalization",(PyCFunction)PyDataSet_disable_normalization, METH_VARARGS, "disable normalization" },

    {NULL}  /* Sentinel */
};

#pragma endregion
#pragma region Solver
// ---------------------------- Initializers ------------------------------
static int PySolver_init(PySolver *self, PyObject *args, PyObject *kwds)
{
    char* type;
    // Parse the input argument to get the type as a C string
    if (!PyArg_ParseTuple(args, "s", &type)) {
        return -1;
    }

    self->type = std::string(type);
    // Initialize the ptrObj based on the type
    if (strcmp(type, "IQHV") == 0) {
        self->ptrObj = new IQHVSolver();
    }
    else if (strcmp(type, "HSS") == 0) {
        self->ptrObj = new HSSSolver();
    }
    else if (strcmp(type, "HVE") == 0) {
        self->ptrObj = new HVE();
    }
    else if (strcmp(type, "MCHV") == 0) {
        self->ptrObj = new MCHV(); 
    }
    else if (strcmp(type, "QEHC") == 0) {
        self->ptrObj = new QEHC(); 
    }
    else if (strcmp(type, "QHV_BQ") == 0) {
        self->ptrObj = new QHV_BQ(); 
    }
    else if (strcmp(type, "QHV_BR") == 0) {
        self->ptrObj = new QHV_BR(); 
    }
    else if (strcmp(type, "QR2") == 0) {
        self->ptrObj = new QR2(); 
    }
    else {
        std::cout << "Unknown solver type: " << type << std::endl;
        return -1; // Return -1 to indicate error with invalid type
    }

    // Set default callback functions
    self->ptrObj->StartCallback = &DefaultStartCallback;
    self->ptrObj->EndCallback = &DefaultEndCallback;
    self->ptrObj->IterationCallback = &ProgressBarCallback;

    return 0;
}


// ------------------------- destructors/deallocators --------------------------
static void PySolver_dealloc(PySolver * self)
// destruct the object
{
    delete self->ptrObj;
    Py_TYPE(self)->tp_free(self);
}


// ----------------------- types definitions ---------------------------------
static PyTypeObject PySolverType = { PyVarObject_HEAD_INIT(NULL, 0)
    "moda.Solver"   /* tp_name */
    
};
// ---------------------------- Solver methods --------------------------------------

static PyObject * PySolver_SolveDirectoryBulk(PySolver* self, PyObject* args)
{
    char* fn;
    if (! PyArg_ParseTuple(args, "z*", &fn))
         return Py_False;
    auto all_problems = DataSet::LoadBulk(fn, false);
    Result* res = (self->ptrObj)->Solve(&all_problems[0]);
    HypervolumeResult* hvres;


    int iterator = 0;
    PyObject* pyResultList = PyList_New(all_problems.size());
    for (auto problem : all_problems)
    { 
        SolverSettings settings = SolverSettings(problem, SolverSettings::zeroone);
        res = (self->ptrObj)->Solve(&problem, settings);
        hvres =  (HypervolumeResult*)res;
        PyObject* pyResult = Py_BuildValue("d", hvres->Volume);
        PyList_SetItem(pyResultList, iterator, pyResult);
        iterator++;

    }
    return pyResultList;
}

static PyObject * PySolver_SolveFile(PySolver* self, PyObject* args)
{
    char* fn;
    if (! PyArg_ParseTuple(args, "z*", &fn))
         return Py_False;
    auto problem = DataSet::DataSet::LoadFromFilename(fn, false);
    Result* res;
    HypervolumeResult* hvres;


    SolverSettings settings = SolverSettings(problem, SolverSettings::zeroone);
    res = (self->ptrObj)->Solve(&problem, settings);
    hvres =  (HypervolumeResult*)res;
    return Py_BuildValue("d",hvres->Volume);
}

static PyObject * PySolver_Solve(PySolver* self, PyObject* args)
{
    PyObject* obj, *solver_settings_obj = NULL;
    if (!PyArg_ParseTuple(args, "O|O", &obj, &solver_settings_obj)) {
        PyErr_SetString(PyExc_TypeError, "Expected a DataSet object");
        return NULL;
    }

    // Ensure obj is of type PyDataSet
    if (!PyObject_TypeCheck(obj, &PyDataSetType)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be a DataSet object");
        return NULL;
    }

    PyDataSet* dso = (PyDataSet*)obj;
    
    if (dso->ptrObj == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "DataSet object is uninitialized");
        return NULL;
    }
    
    SolverSettings settings;
    if (solver_settings_obj == NULL) {
        std::cout << "here";
        settings = SolverSettings(*(dso->ptrObj), SolverSettings::zeroone);
        settings.SubsetSelectionQHVIncremental = false;
    }
    else
    {
        
        settings = *((PySolverSettingsObject*)solver_settings_obj)->settings;
    }
    
    Result* res = (self->ptrObj)->Solve((dso->ptrObj), settings);
    
    if (!res) {
        PyErr_SetString(PyExc_RuntimeError, "Solver returned a null result");
        return NULL;
    }
    PyHypervolumeResultObject* result_hv;
    PySubsetSelectionResultObject* result_hss;
    PyBoundedResultObject* result_hve;
    PyBoundedResultObject* result_mchv;
    PyContributionResultObject* result_qehc;
    PyBoundedResultObject* result_bq;
    PyBoundedResultObject* result_br;
    PyR2ResultObject* result_r2;
    if (strcmp(self->type.c_str(), "IQHV") == 0) {

        result_hv = PyObject_New(PyHypervolumeResultObject,&PyHypervolumeResultType);
        auto hvr = (HypervolumeResult*)res;
        result_hv->result = new HypervolumeResult();
        result_hv->result->Volume = hvr->Volume;
        result_hv->base.result = new Result();
        result_hv->base.result->ElapsedTime = hvr->ElapsedTime;
        result_hv->base.result->FinalResult = hvr->FinalResult;
        return (PyObject*)result_hv;
    }
    else if (strcmp(self->type.c_str(), "HSS") == 0) {
        result_hss = PyObject_New(PySubsetSelectionResultObject,&PySubsetSelectionResultType);
        auto hvr = (SubsetSelectionResult*)res;
        result_hss->result = new SubsetSelectionResult();
        result_hss->result->selectedPoints = hvr->selectedPoints;
        result_hss->result->Volume = hvr->Volume;
        result_hss->base.result = new Result();
        result_hss->base.result->ElapsedTime = hvr->ElapsedTime;
        result_hss->base.result->FinalResult = hvr->FinalResult;
        return (PyObject*)result_hss;
    }
    else if (strcmp(self->type.c_str(), "HVE") == 0) {
        result_hve = PyObject_New(PyBoundedResultObject,&PyBoundedResultType);
        auto hvr = (BoundedResult*)res;
        result_hve->result = new BoundedResult();
        result_hve->result->Guaranteed = hvr->Guaranteed;
        result_hve->result->LowerBound = hvr->LowerBound;
        result_hve->result->UpperBound = hvr->UpperBound;
        result_hve->result->VolumeEstimation = hvr->VolumeEstimation;
        result_hve->base.result = new Result();
        result_hve->base.result->ElapsedTime = hvr->ElapsedTime;
        result_hve->base.result->FinalResult = hvr->FinalResult;
        return (PyObject*)result_hve;
    }
    else if (strcmp(self->type.c_str(), "MCHV") == 0) {
        result_mchv = PyObject_New(PyBoundedResultObject,&PyBoundedResultType);
        auto hvr = (BoundedResult*)res;
        result_mchv->result = new BoundedResult();
        result_mchv->result->Guaranteed = hvr->Guaranteed;
        result_mchv->result->LowerBound = hvr->LowerBound;
        result_mchv->result->UpperBound = hvr->UpperBound;
        result_mchv->result->VolumeEstimation = hvr->VolumeEstimation;
        result_mchv->base.result = new Result();
        result_mchv->base.result->ElapsedTime = hvr->ElapsedTime;
        result_mchv->base.result->FinalResult = hvr->FinalResult;
        return (PyObject*)result_mchv; 
    }
    else if (strcmp(self->type.c_str(), "QEHC") == 0) {
        result_qehc = PyObject_New(PyContributionResultObject,&PyContributionResultType);
        auto hvr = (ContributionResult*)res;
        result_qehc->result = new ContributionResult();
        result_qehc->result->MaximumContribution = hvr->MaximumContribution;
        result_qehc->result->MinimumContribution = hvr->MinimumContribution;
        result_qehc->result->MaximumContributionIndex = hvr->MaximumContributionIndex;
        result_qehc->result->MinimumContributionIndex = hvr->MinimumContributionIndex;
        result_qehc->base.result = new Result();
        result_qehc->base.result->ElapsedTime = hvr->ElapsedTime;
        result_qehc->base.result->FinalResult = hvr->FinalResult;
        return (PyObject*)result_qehc;
    }
    else if (strcmp(self->type.c_str(), "QHV_BQ") == 0) {
        result_bq = PyObject_New(PyBoundedResultObject,&PyBoundedResultType);
        auto hvr = (BoundedResult*)res;
        result_bq->result = new BoundedResult();
        result_bq->result->Guaranteed = hvr->Guaranteed;
        result_bq->result->LowerBound = hvr->LowerBound;
        result_bq->result->UpperBound = hvr->UpperBound;
        result_bq->result->VolumeEstimation = hvr->VolumeEstimation;
        result_bq->base.result = new Result();
        result_bq->base.result->ElapsedTime = hvr->ElapsedTime;
        result_bq->base.result->FinalResult = hvr->FinalResult;
        return (PyObject*)result_bq; 
    }
    else if (strcmp(self->type.c_str(), "QHV_BR") == 0) {
        result_br = PyObject_New(PyBoundedResultObject,&PyBoundedResultType);
        auto hvr = (BoundedResult*)res;
        result_br->result = new BoundedResult();
        result_br->result->Guaranteed = hvr->Guaranteed;
        result_br->result->LowerBound = hvr->LowerBound;
        result_br->result->UpperBound = hvr->UpperBound;
        result_br->result->VolumeEstimation = hvr->VolumeEstimation;
        result_br->base.result = new Result();
        result_br->base.result->ElapsedTime = hvr->ElapsedTime;
        result_br->base.result->FinalResult = hvr->FinalResult;
        return (PyObject*)result_br;
    }
    else if (strcmp(self->type.c_str(), "QR2") == 0) {
        result_r2 = PyObject_New(PyR2ResultObject,&PyR2ResultType);
        auto hvr = (R2Result*)res;
        result_r2->result = new R2Result();
        result_r2->result->Hypervolume = hvr->Hypervolume;
        result_r2->result->R2 = hvr->R2;

        result_r2->base.result = new Result();
        result_r2->base.result->ElapsedTime = hvr->ElapsedTime;
        result_r2->base.result->FinalResult = hvr->FinalResult;
        return (PyObject*)result_r2; 
    }
    else {
        std::cout << "Unknown solver type: " << self->type << std::endl;
        return Py_None; // Return -1 to indicate error with invalid type
    }

}

static PyMethodDef PySolver_methods[] = {
    { "solve_directory_bulk", (PyCFunction)PySolver_SolveDirectoryBulk,    METH_VARARGS,       "calculate hypervolume value of all solutions in a given directory" },
    { "solve_file", (PyCFunction)PySolver_SolveFile,    METH_VARARGS,       "calculate hypervolume value of the solution in a given file" },
    { "solve", (PyCFunction)PySolver_Solve,    METH_VARARGS,       "calculate hypervolume value of a given dataset" },

    {NULL}  /* Sentinel */
};
#pragma endregion
#pragma region callback holders
// Global Python callback holders
static PyObject* pyStartCallback = NULL;
static PyObject* pyIterationCallback = NULL;
static PyObject* pyEndCallback = NULL;

// C++ Wrappers for Solver Callbacks
void StartCallbackWrapper(DataSetParameters problemSettings, std::string SolverMessage) {
    if (pyStartCallback && PyCallable_Check(pyStartCallback)) {
        PyGILState_STATE gstate = PyGILState_Ensure();  // Ensure thread safety
        PyObject_CallFunction(pyStartCallback, "s", SolverMessage.c_str());
        PyGILState_Release(gstate);
    }
}

void IterationCallbackWrapper(int currentIteration, int totalIterations, Result* stepResult) {
    if (pyIterationCallback && PyCallable_Check(pyIterationCallback)) {
        PyGILState_STATE gstate = PyGILState_Ensure();
        
        if (stepResult->type == Result::Hypervolume) {
            PyHypervolumeResultObject* pyStepResult = PyObject_New(PyHypervolumeResultObject, &PyHypervolumeResultType);
            auto hvr = (HypervolumeResult*) stepResult;
            pyStepResult->result = new HypervolumeResult();
            pyStepResult->result->Volume = hvr->Volume;
            pyStepResult->base.result = new Result();
            pyStepResult->base.result->ElapsedTime = hvr->ElapsedTime;
            pyStepResult->base.result->FinalResult = hvr->FinalResult;
            PyObject_CallFunction(pyIterationCallback, "iiO", currentIteration, totalIterations, pyStepResult);
        } 

        if (stepResult->type == Result::Estimation) {
            PyBoundedResultObject* pyStepResult = PyObject_New(PyBoundedResultObject, &PyBoundedResultType);
            auto hvr = (BoundedResult*) stepResult;
            pyStepResult->result = new BoundedResult();
            pyStepResult->result->Guaranteed = hvr->Guaranteed;
            pyStepResult->result->LowerBound = hvr->LowerBound;
            pyStepResult->result->UpperBound = hvr->UpperBound;
            pyStepResult->result->VolumeEstimation = hvr->VolumeEstimation;
            pyStepResult->base.result = new Result();
            pyStepResult->base.result->ElapsedTime = hvr->ElapsedTime;
            pyStepResult->base.result->FinalResult = hvr->FinalResult;
            PyObject_CallFunction(pyIterationCallback, "iiO", currentIteration, totalIterations, pyStepResult);
        } 

        if (stepResult->type == Result::Contribution) {
            PyContributionResultObject* pyStepResult = PyObject_New(PyContributionResultObject, &PyContributionResultType);
            auto hvr = (ContributionResult*) stepResult;

            pyStepResult->result = new ContributionResult();
            pyStepResult->result->MaximumContribution = hvr->MaximumContribution;
            pyStepResult->result->MinimumContribution = hvr->MinimumContribution;
            pyStepResult->result->MaximumContributionIndex = hvr->MaximumContributionIndex;
            pyStepResult->result->MinimumContributionIndex = hvr->MinimumContributionIndex;
            pyStepResult->base.result = new Result();
            pyStepResult->base.result->ElapsedTime = hvr->ElapsedTime;
            pyStepResult->base.result->FinalResult = hvr->FinalResult;
            PyObject_CallFunction(pyIterationCallback, "iiO", currentIteration, totalIterations, pyStepResult);
        } 

        if (stepResult->type == Result::SubsetSelection) {
            PySubsetSelectionResultObject* pyStepResult = PyObject_New(PySubsetSelectionResultObject, &PySubsetSelectionResultType);
            auto hvr = (SubsetSelectionResult*) stepResult;
            pyStepResult->result = new SubsetSelectionResult();
            pyStepResult->result->selectedPoints = hvr->selectedPoints;
            pyStepResult->result->Volume = hvr->Volume;
            pyStepResult->base.result = new Result();
            pyStepResult->base.result->ElapsedTime = hvr->ElapsedTime;
            pyStepResult->base.result->FinalResult = hvr->FinalResult;
            PyObject_CallFunction(pyIterationCallback, "iiO", currentIteration, totalIterations, pyStepResult);
        } 

        if (stepResult->type == Result::R2) {
            PyR2ResultObject* pyStepResult = PyObject_New(PyR2ResultObject, &PyR2ResultType);
            auto hvr = (R2Result*) stepResult;
            pyStepResult->result = new R2Result();
            pyStepResult->result->Hypervolume = hvr->Hypervolume;
            pyStepResult->result->R2 = hvr->R2;
            pyStepResult->base.result = new Result();
            pyStepResult->base.result->ElapsedTime = hvr->ElapsedTime;
            pyStepResult->base.result->FinalResult = hvr->FinalResult;
            PyObject_CallFunction(pyIterationCallback, "iiO", currentIteration, totalIterations, pyStepResult);
        } 
        PyGILState_Release(gstate);
    }
}

void EndCallbackWrapper(DataSetParameters problemSettings, Result* finalResult) {
    if (pyEndCallback && PyCallable_Check(pyEndCallback)) {
        PyGILState_STATE gstate = PyGILState_Ensure();


        if (finalResult->type == Result::Hypervolume) {
            PyHypervolumeResultObject* pyFinalResult = PyObject_New(PyHypervolumeResultObject, &PyHypervolumeResultType);
            auto hvr = (HypervolumeResult*) finalResult;
            pyFinalResult->result = new HypervolumeResult();
            pyFinalResult->result->Volume = hvr->Volume;
            pyFinalResult->base.result = new Result();
            pyFinalResult->base.result->ElapsedTime = hvr->ElapsedTime;
            pyFinalResult->base.result->FinalResult = hvr->FinalResult;
            PyObject_CallFunction(pyEndCallback, "O", pyFinalResult);
        } 

        if (finalResult->type == Result::Estimation) {
            PyBoundedResultObject* pyFinalResult = PyObject_New(PyBoundedResultObject, &PyBoundedResultType);
            auto hvr = (BoundedResult*) finalResult;
            pyFinalResult->result = new BoundedResult();
            pyFinalResult->result->Guaranteed = hvr->Guaranteed;
            pyFinalResult->result->LowerBound = hvr->LowerBound;
            pyFinalResult->result->UpperBound = hvr->UpperBound;
            pyFinalResult->result->VolumeEstimation = hvr->VolumeEstimation;
            pyFinalResult->base.result = new Result();
            pyFinalResult->base.result->ElapsedTime = hvr->ElapsedTime;
            pyFinalResult->base.result->FinalResult = hvr->FinalResult;
            PyObject_CallFunction(pyEndCallback, "O", pyFinalResult);
        } 

        if (finalResult->type == Result::Contribution) {
            PyContributionResultObject* pyFinalResult = PyObject_New(PyContributionResultObject, &PyContributionResultType);
            auto hvr = (ContributionResult*) finalResult;

            pyFinalResult->result = new ContributionResult();
            pyFinalResult->result->MaximumContribution = hvr->MaximumContribution;
            pyFinalResult->result->MinimumContribution = hvr->MinimumContribution;
            pyFinalResult->result->MaximumContributionIndex = hvr->MaximumContributionIndex;
            pyFinalResult->result->MinimumContributionIndex = hvr->MinimumContributionIndex;
            pyFinalResult->base.result = new Result();
            pyFinalResult->base.result->ElapsedTime = hvr->ElapsedTime;
            pyFinalResult->base.result->FinalResult = hvr->FinalResult;
            PyObject_CallFunction(pyEndCallback, "O", pyFinalResult);
        } 

        if (finalResult->type == Result::SubsetSelection) {
            PySubsetSelectionResultObject* pyFinalResult = PyObject_New(PySubsetSelectionResultObject, &PySubsetSelectionResultType);
            auto hvr = (SubsetSelectionResult*) finalResult;
            pyFinalResult->result = new SubsetSelectionResult();
            pyFinalResult->result->selectedPoints = hvr->selectedPoints;
            pyFinalResult->result->Volume = hvr->Volume;
            pyFinalResult->base.result = new Result();
            pyFinalResult->base.result->ElapsedTime = hvr->ElapsedTime;
            pyFinalResult->base.result->FinalResult = hvr->FinalResult;
            PyObject_CallFunction(pyEndCallback, "O", pyFinalResult);
        } 

        if (finalResult->type == Result::R2) {
            PyR2ResultObject* pyFinalResult = PyObject_New(PyR2ResultObject, &PyR2ResultType);
            auto hvr = (R2Result*) finalResult;
            pyFinalResult->result = new R2Result();
            pyFinalResult->result->Hypervolume = hvr->Hypervolume;
            pyFinalResult->result->R2 = hvr->R2;
            pyFinalResult->base.result = new Result();
            pyFinalResult->base.result->ElapsedTime = hvr->ElapsedTime;
            pyFinalResult->base.result->FinalResult = hvr->FinalResult;
            PyObject_CallFunction(pyEndCallback, "O", pyFinalResult);
        } 
        PyGILState_Release(gstate);
    }
}

// Set Start Callback
static PyObject* py_set_start_callback(PyObject* self, PyObject* args) {
    PyObject* callback;
    if (!PyArg_ParseTuple(args, "O", &callback)) {
        return NULL;
    }

    if (!PyCallable_Check(callback)) {
        PyErr_SetString(PyExc_TypeError, "Callback must be callable");
        return NULL;
    }

    Py_XINCREF(callback);
    Py_XDECREF(pyStartCallback);
    pyStartCallback = callback;

    Py_RETURN_NONE;
}

// Set Iteration Callback
static PyObject* py_set_iteration_callback(PyObject* self, PyObject* args) {
    PyObject* callback;
    if (!PyArg_ParseTuple(args, "O", &callback)) {
        return NULL;
    }

    if (!PyCallable_Check(callback)) {
        PyErr_SetString(PyExc_TypeError, "Callback must be callable");
        return NULL;
    }

    Py_XINCREF(callback);
    Py_XDECREF(pyIterationCallback);
    pyIterationCallback = callback;

    Py_RETURN_NONE;
}

// Set End Callback
static PyObject* py_set_end_callback(PyObject* self, PyObject* args) {
    PyObject* callback;
    if (!PyArg_ParseTuple(args, "O", &callback)) {
        return NULL;
    }

    if (!PyCallable_Check(callback)) {
        PyErr_SetString(PyExc_TypeError, "Callback must be callable");
        return NULL;
    }

    Py_XINCREF(callback);
    Py_XDECREF(pyEndCallback);
    pyEndCallback = callback;

    Py_RETURN_NONE;
}

static PyObject* py_assign_callbacks(PyObject* self, PyObject* args) {
    PySolver* capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)) {
        return NULL;
    }

    qhv::Solver* solver = (Solver*)capsule->ptrObj;
    if (!solver) {
        PyErr_SetString(PyExc_RuntimeError, "Invalid Solver Object");
        return NULL;
    }

    // Assign function pointers to the global wrappers
    solver->StartCallback = StartCallbackWrapper;
    solver->IterationCallback = IterationCallbackWrapper;
    solver->EndCallback = EndCallbackWrapper;

    Py_RETURN_NONE;
}

// Method table for the Python module
static PyMethodDef CallbackMethods[] = {
    {"set_start_callback", py_set_start_callback, METH_VARARGS, "Set the start callback"},
    {"set_iteration_callback", py_set_iteration_callback, METH_VARARGS, "Set the iteration callback"},
    {"set_end_callback", py_set_end_callback, METH_VARARGS, "Set the end callback"},
    {"assign_callbacks", py_assign_callbacks, METH_VARARGS, "Assign callbacks to an existing Solver"},
    {NULL, NULL, 0, NULL}  // Sentinel
};




#pragma endregion
#pragma region module definiton

static PyModuleDef modamodule = {
    PyModuleDef_HEAD_INIT,
    "moda",
    "module definition",
    -1,
    CallbackMethods, NULL, NULL, NULL, NULL
};

#pragma endregion
#pragma region package initializer
// -------------------------- package initializer -------------------------------
PyMODINIT_FUNC PyInit_moda(void)
// create the module
{
    import_array();
    PyObject* moduleObject;
    //qhv::IQHV c;
    PySolverType.tp_new = PyType_GenericNew;
    PySolverType.tp_basicsize=sizeof(PySolver);
    PySolverType.tp_dealloc=(destructor) PySolver_dealloc;
    PySolverType.tp_flags=Py_TPFLAGS_DEFAULT;
    PySolverType.tp_doc="Solver objects";
    PySolverType.tp_methods=PySolver_methods;
    PySolverType.tp_init=(initproc)PySolver_init;

    //qhv::Point p;
    PyPointType.tp_new = PyType_GenericNew;
    PyPointType.tp_basicsize=sizeof(PyPoint);
    PyPointType.tp_dealloc=(destructor) PyPoint_dealloc;
    PyPointType.tp_flags=Py_TPFLAGS_DEFAULT;
    PyPointType.tp_doc="Point objects";
    PyPointType.tp_methods=PyPoint_methods;
    PyPointType.tp_init = (initproc)PyPoint_init;
    PyPointType.tp_as_mapping = &PyPoint_mapping;
    PyPointType.tp_str = PyPoint_toString;
    
    PyDataSetType.tp_new = PyType_GenericNew;
    PyDataSetType.tp_basicsize=sizeof(PyDataSet);
    PyDataSetType.tp_dealloc=(destructor) PyDataSet_dealloc;
    PyDataSetType.tp_flags=Py_TPFLAGS_DEFAULT;
    PyDataSetType.tp_doc="Dataset objects";
    PyDataSetType.tp_methods=PyDataSet_methods;
    PyDataSetType.tp_init = (initproc)PyDataSet_init;
    PyDataSetType.tp_as_mapping = &PyDataSet_mapping;
    PyDataSetType.tp_getset = PyDataSet_getseters;
    PyDataSetType.tp_str = PyDataSet_toString;




    if (PyType_Ready(&PySolverType) < 0)
        return NULL;

    if (PyType_Ready(&PyPointType) < 0)
        return NULL;

    if (PyType_Ready(&PyDataSetType) < 0)
        return NULL;

    if (PyType_Ready(&PyResultType) < 0)
        return NULL;

    if (PyType_Ready(&PyHypervolumeResultType) < 0)
        return NULL;
    if (PyType_Ready(&PySubsetSelectionResultType) < 0)
        return NULL;
    if (PyType_Ready(&PyBoundedResultType) < 0)
        return NULL;
    if (PyType_Ready(&PyR2ResultType) < 0)
        return NULL;    
    if (PyType_Ready(&PyContributionResultType) < 0)
        return NULL;   
    if (PyType_Ready(&PyDataSetParametersType) < 0)
        return NULL;
    if (PyType_Ready(&PySwitchSettingsType) < 0)
        return NULL;
    if (PyType_Ready(&PySolverSettingsType) < 0)
        return NULL;
    moduleObject = PyModule_Create(&modamodule);
    if (moduleObject == NULL)
        return NULL;


    Py_INCREF(&PySolverType);
    Py_INCREF(&PyPointType);
    Py_INCREF(&PyDataSetType);
    Py_INCREF(&PyResultType);
    Py_INCREF(&PyHypervolumeResultType);
    Py_INCREF(&PySubsetSelectionResultType);
    Py_INCREF(&PyBoundedResultType);
    Py_INCREF(&PyR2ResultType);
    Py_INCREF(&PyContributionResultType);
    Py_INCREF(&PyDataSetParametersType);
    Py_INCREF(&PySwitchSettingsType);
    Py_INCREF(&PySolverSettingsType);

    PyModule_AddObject(moduleObject, "Solver", (PyObject *)&PySolverType);
    PyModule_AddObject(moduleObject, "Point", (PyObject *)&PyPointType);
    PyModule_AddObject(moduleObject, "DataSet", (PyObject *)&PyDataSetType);
    PyModule_AddObject(moduleObject, "Result", (PyObject*)&PyResultType);
    PyModule_AddObject(moduleObject, "HypervolumeResult", (PyObject*)&PyHypervolumeResultType);
    PyModule_AddObject(moduleObject, "SubsetSelectionResult", (PyObject*)&PySubsetSelectionResultType);
    PyModule_AddObject(moduleObject, "BoundedResult", (PyObject*)&PyBoundedResultType);
    PyModule_AddObject(moduleObject, "R2Result", (PyObject*)&PyR2ResultType);
    PyModule_AddObject(moduleObject, "ContributionResult", (PyObject*)&PyContributionResultType);
    PyModule_AddObject(moduleObject, "DataSetParameters", (PyObject*)&PyDataSetParametersType);
    PyModule_AddObject(moduleObject, "SwitchSettings", (PyObject*)&PySwitchSettingsType);
    PyModule_AddObject(moduleObject, "SolverSettings", (PyObject*)&PySolverSettingsType);

    PyObject* ComparisonResultEnum = PyDict_New();
    PyDict_SetItemString(ComparisonResultEnum, "Dominating", Py_BuildValue("i",ComparisonResult::_Dominating));
    PyDict_SetItemString(ComparisonResultEnum, "Dominated", Py_BuildValue("i",ComparisonResult::_Dominated));
    PyDict_SetItemString(ComparisonResultEnum, "Nondominated", Py_BuildValue("i",ComparisonResult::_Nondominated));
    PyDict_SetItemString(ComparisonResultEnum, "EqualSol", Py_BuildValue("i",ComparisonResult::_EqualSol));

    Py_INCREF(ComparisonResultEnum);  // Prevent early deallocation
    PyModule_AddObject(moduleObject, "ComparisonResult", ComparisonResultEnum);

    PyObject* moduleDict = PyImport_GetModuleDict();
    PyDict_SetItemString(moduleDict, "moda.Point", (PyObject *)&PyPointType);
    PyDict_SetItemString(moduleDict, "moda.DataSet", (PyObject *)&PyDataSetType);
    PyDict_SetItemString(moduleDict, "moda.Settings.DataSetParameters", (PyObject *)&PyDataSetParametersType);
    PyDict_SetItemString(moduleDict, "moda.Settings.SwitchSetting", (PyObject *)&PySwitchSettingsType);
    PyDict_SetItemString(moduleDict, "moda.Settings.SolverSettings", (PyObject *)&PySolverSettingsType);
    
    return moduleObject;
}
#pragma endregion
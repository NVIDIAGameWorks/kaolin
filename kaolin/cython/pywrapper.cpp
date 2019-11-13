// Copyright 2019 Lars Mescheder, Michael Oechsle,
// Michael Niemeyer, Andreas Geiger, Sebastian Nowozin

// Permission is hereby granted, free of charge,
// to any person obtaining a copy of this software and
// associated documentation files (the "Software"), to 
//  in the Software without restriction, including without
//  limitation the rights to use, copy, modify, merge, publish,
//  distribute, sublicense, and/or sell copies of the Software,
//  and to permit persons to whom the Software is furnished to do so,
//  subject to the following conditions:

// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
// BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "pywrapper.h"

#include "marchingcubes.h"

#include <stdexcept>

struct PythonToCFunc
{
    PyObject* func;
    PythonToCFunc(PyObject* func) {this->func = func;}
    double operator()(double x, double y, double z)
    {
        PyObject* res = PyObject_CallFunction(func, "(d,d,d)", x, y, z); // py::extract<double>(func(x,y,z));
        if(res == NULL)
            return 0.0;
        
        double result = PyFloat_AsDouble(res);
        Py_DECREF(res);
        return result;
    }
};

PyObject* marching_cubes_func(PyObject* lower, PyObject* upper,
    int numx, int numy, int numz, PyObject* f, double isovalue)
{
    std::vector<double> vertices;
    std::vector<size_t> polygons;
    
    // Copy the lower and upper coordinates to a C array.
    double lower_[3];
    double upper_[3];
    for(int i=0; i<3; ++i)
    {
        PyObject* l = PySequence_GetItem(lower, i);
        if(l == NULL)
            throw std::runtime_error("error");
        PyObject* u = PySequence_GetItem(upper, i);
        if(u == NULL)
        {
            Py_DECREF(l);
            throw std::runtime_error("error");
        }
        
        lower_[i] = PyFloat_AsDouble(l);
        upper_[i] = PyFloat_AsDouble(u);
        
        Py_DECREF(l);
        Py_DECREF(u);
        if(lower_[i]==-1.0 || upper_[i]==-1.0)
        {
            if(PyErr_Occurred())
                throw std::runtime_error("error");
        }
    }
    
    // Marching cubes.
    mc::marching_cubes<double>(lower_, upper_, numx, numy, numz, PythonToCFunc(f), isovalue, vertices, polygons);
    
    // Copy the result to two Python ndarrays.
    npy_intp size_vertices = vertices.size();
    npy_intp size_polygons = polygons.size();
    PyArrayObject* verticesarr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size_vertices, PyArray_DOUBLE));
    PyArrayObject* polygonsarr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size_polygons, PyArray_ULONG));
    
    std::vector<double>::const_iterator it = vertices.begin();
    for(int i=0; it!=vertices.end(); ++i, ++it)
        *reinterpret_cast<double*>(PyArray_GETPTR1(verticesarr, i)) = *it;
    std::vector<size_t>::const_iterator it2 = polygons.begin();
    for(int i=0; it2!=polygons.end(); ++i, ++it2)
        *reinterpret_cast<unsigned long*>(PyArray_GETPTR1(polygonsarr, i)) = *it2;
    
    PyObject* res = Py_BuildValue("(O,O)", verticesarr, polygonsarr);
    Py_XDECREF(verticesarr);
    Py_XDECREF(polygonsarr);
    return res;
}

struct PyArrayToCFunc
{
    PyArrayObject* arr;
    PyArrayToCFunc(PyArrayObject* arr) {this->arr = arr;}
    double operator()(int x, int y, int z)
    {
        npy_intp c[3] = {x,y,z};
        return PyArray_SafeGet<double>(arr, c);
    }
};

PyObject* marching_cubes(PyArrayObject* arr, double isovalue)
{
    if(PyArray_NDIM(arr) != 3)
        throw std::runtime_error("Only three-dimensional arrays are supported.");
    
    // Prepare data.
    npy_intp* shape = PyArray_DIMS(arr);
    double lower[3] = {0,0,0};
    double upper[3] = {shape[0]-1, shape[1]-1, shape[2]-1};
    long numx = upper[0] - lower[0] + 1;
    long numy = upper[1] - lower[1] + 1;
    long numz = upper[2] - lower[2] + 1;
    std::vector<double> vertices;
    std::vector<size_t> polygons;
    
    // Marching cubes.
    mc::marching_cubes<double>(lower, upper, numx, numy, numz, PyArrayToCFunc(arr), isovalue,
                        vertices, polygons);
    
    // Copy the result to two Python ndarrays.
    npy_intp size_vertices = vertices.size();
    npy_intp size_polygons = polygons.size();
    PyArrayObject* verticesarr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size_vertices, PyArray_DOUBLE));
    PyArrayObject* polygonsarr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size_polygons, PyArray_ULONG));
    
    std::vector<double>::const_iterator it = vertices.begin();
    for(int i=0; it!=vertices.end(); ++i, ++it)
        *reinterpret_cast<double*>(PyArray_GETPTR1(verticesarr, i)) = *it;
    std::vector<size_t>::const_iterator it2 = polygons.begin();
    for(int i=0; it2!=polygons.end(); ++i, ++it2)
        *reinterpret_cast<unsigned long*>(PyArray_GETPTR1(polygonsarr, i)) = *it2;
    
    PyObject* res = Py_BuildValue("(O,O)", verticesarr, polygonsarr);
    Py_XDECREF(verticesarr);
    Py_XDECREF(polygonsarr);
    
    return res;
}

PyObject* marching_cubes2(PyArrayObject* arr, double isovalue)
{
    if(PyArray_NDIM(arr) != 3)
        throw std::runtime_error("Only three-dimensional arrays are supported.");

    // Prepare data.
    npy_intp* shape = PyArray_DIMS(arr);
    double lower[3] = {0,0,0};
    double upper[3] = {shape[0]-1, shape[1]-1, shape[2]-1};
    long numx = upper[0] - lower[0] + 1;
    long numy = upper[1] - lower[1] + 1;
    long numz = upper[2] - lower[2] + 1;
    std::vector<double> vertices;
    std::vector<size_t> polygons;

    // Marching cubes.
    mc::marching_cubes2<double>(lower, upper, numx, numy, numz, PyArrayToCFunc(arr), isovalue,
                        vertices, polygons);

    // Copy the result to two Python ndarrays.
    npy_intp size_vertices = vertices.size();
    npy_intp size_polygons = polygons.size();
    PyArrayObject* verticesarr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size_vertices, PyArray_DOUBLE));
    PyArrayObject* polygonsarr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size_polygons, PyArray_ULONG));

    std::vector<double>::const_iterator it = vertices.begin();
    for(int i=0; it!=vertices.end(); ++i, ++it)
        *reinterpret_cast<double*>(PyArray_GETPTR1(verticesarr, i)) = *it;
    std::vector<size_t>::const_iterator it2 = polygons.begin();
    for(int i=0; it2!=polygons.end(); ++i, ++it2)
        *reinterpret_cast<unsigned long*>(PyArray_GETPTR1(polygonsarr, i)) = *it2;

    PyObject* res = Py_BuildValue("(O,O)", verticesarr, polygonsarr);
    Py_XDECREF(verticesarr);
    Py_XDECREF(polygonsarr);

    return res;
}

PyObject* marching_cubes3(PyArrayObject* arr, double isovalue)
{
    if(PyArray_NDIM(arr) != 3)
        throw std::runtime_error("Only three-dimensional arrays are supported.");

    // Prepare data.
    npy_intp* shape = PyArray_DIMS(arr);
    double lower[3] = {0,0,0};
    double upper[3] = {shape[0]-1, shape[1]-1, shape[2]-1};
    long numx = upper[0] - lower[0] + 1;
    long numy = upper[1] - lower[1] + 1;
    long numz = upper[2] - lower[2] + 1;
    std::vector<double> vertices;
    std::vector<size_t> polygons;

    // Marching cubes.
    mc::marching_cubes3<double>(lower, upper, numx, numy, numz, PyArrayToCFunc(arr), isovalue,
                        vertices, polygons);

    // Copy the result to two Python ndarrays.
    npy_intp size_vertices = vertices.size();
    npy_intp size_polygons = polygons.size();
    PyArrayObject* verticesarr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size_vertices, PyArray_DOUBLE));
    PyArrayObject* polygonsarr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size_polygons, PyArray_ULONG));

    std::vector<double>::const_iterator it = vertices.begin();
    for(int i=0; it!=vertices.end(); ++i, ++it)
        *reinterpret_cast<double*>(PyArray_GETPTR1(verticesarr, i)) = *it;
    std::vector<size_t>::const_iterator it2 = polygons.begin();
    for(int i=0; it2!=polygons.end(); ++i, ++it2)
        *reinterpret_cast<unsigned long*>(PyArray_GETPTR1(polygonsarr, i)) = *it2;

    PyObject* res = Py_BuildValue("(O,O)", verticesarr, polygonsarr);
    Py_XDECREF(verticesarr);
    Py_XDECREF(polygonsarr);

    return res;
}
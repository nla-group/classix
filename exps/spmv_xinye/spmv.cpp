#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject* spsubmatxvec(PyObject* self, PyObject* args) {
    PyArrayObject *m_data_arr, *m_indptr_arr, *m_indices_arr;
    PyArrayObject *v_data_arr, *result_arr;
    int start_row, end_row;

    if (!PyArg_ParseTuple(args, "O!O!O!iiO!O!",
                          &PyArray_Type, &m_data_arr,
                          &PyArray_Type, &m_indptr_arr,
                          &PyArray_Type, &m_indices_arr,
                          &start_row, &end_row,
                          &PyArray_Type, &v_data_arr,
                          &PyArray_Type, &result_arr)) {
        return NULL;
    }

    if (PyArray_TYPE(m_data_arr) != NPY_DOUBLE || PyArray_NDIM(m_data_arr) != 1 ||
        PyArray_TYPE(m_indptr_arr) != NPY_INT32 || PyArray_NDIM(m_indptr_arr) != 1 ||
        PyArray_TYPE(m_indices_arr) != NPY_INT32 || PyArray_NDIM(m_indices_arr) != 1 ||
        PyArray_TYPE(v_data_arr) != NPY_DOUBLE || PyArray_NDIM(v_data_arr) != 1 ||
        PyArray_TYPE(result_arr) != NPY_DOUBLE || PyArray_NDIM(result_arr) != 1) {
        PyErr_SetString(PyExc_TypeError, "Invalid array types or dimensions");
        return NULL;
    }

    double *m_data = (double*)PyArray_DATA(m_data_arr);
    int *m_indptr = (int*)PyArray_DATA(m_indptr_arr);
    int *m_indices = (int*)PyArray_DATA(m_indices_arr);
    double *v_data = (double*)PyArray_DATA(v_data_arr);
    double *result = (double*)PyArray_DATA(result_arr);

    for (int i = start_row; i < end_row; i++) {
        int row_start = m_indptr[i];
        int row_end = m_indptr[i + 1];
        double val = 0.0;
        for (int j = row_start; j < row_end; j++) {
            int col = m_indices[j];
            val += m_data[j] * v_data[col];
        }
        result[i - start_row] = val;
    }

    Py_RETURN_NONE;
}

static PyMethodDef SpmvMethods[] = {
    {"spsubmatxvec", spsubmatxvec, METH_VARARGS, "Sparse submatrix-vector multiplication (CSR format)"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef spmvmodule = {
    PyModuleDef_HEAD_INIT,
    "spmv",   
    NULL,
    -1,
    SpmvMethods
};

PyMODINIT_FUNC PyInit_spmv(void) {
    PyObject *m = PyModule_Create(&spmvmodule);
    if (m == NULL) return NULL;
    import_array();  
    return m;
}
#define PY_SSIZE_T_CLEAN

#include <python3.10/Python.h>
#include <python3.10/structmember.h>
#include <python3.10/modsupport.h>
#include <python3.10/moduleobject.h>
#include <python3.10/pycapsule.h>

#define NPY_NO_DEPRECATED_API NPY_1_14_API_VERSION
#include <numpy/ndarrayobject.h>

#include <climits>
#include <stdio.h>
#include <stdint.h>

#include "../lib/cmem/cmem.h"
#include "../lib/umem/umem.h"
#include "../lib/fromloader/fromloader.h"
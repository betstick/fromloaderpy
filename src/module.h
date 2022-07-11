#define PY_SSIZE_T_CLEAN

#pragma once
#include "flver2/flver2.h"
#include "mtd/mtd.h"

static PyMethodDef fromloaderMethods[] = {
	flverMethods[0],
	flverMethods[1],
	//flverMethods[2],
	//mtdMethods[0],
	//{"get_faceset", (PyCFunction) flverGetFaceset, METH_VARARGS, "Returns tris for faces"},
	//{"get_vertex_data", (PyCFunction) flverGetVertData, METH_VARARGS, "Returns all vertex data"},
	//{"get_vertex_data_ordered", (PyCFunction) flverGetVertDataOrdered, METH_VARARGS, "Returns all vertex data"},
	//{"export_gltf", (PyCFunction) flverExportGLTF, METH_NOARGS, "Returns all vertex data"},
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef fromloaderModuleDef = {
	PyModuleDef_HEAD_INIT,
	"fromloader",
	"flver module stuff i don't know",
	-1,
	fromloaderMethods,
	NULL,
	NULL,
	NULL,
	NULL
};

PyMODINIT_FUNC
PyInit_fromloader(void);
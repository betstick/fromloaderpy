#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_14_API_VERSION
#include "module.h"

PyMethodDef fromloaderMethods[] = {
	flverMethods[0],
	flverMethods[1],
	flverMethods[3],
	utilMethods[0],
	//flverMethods[2],
	//mtdMethods[0],
	//{"get_faceset", (PyCFunction) flverGetFaceset, METH_VARARGS, "Returns tris for faces"},
	//{"get_vertex_data", (PyCFunction) flverGetVertData, METH_VARARGS, "Returns all vertex data"},
	//{"get_vertex_data_ordered", (PyCFunction) flverGetVertDataOrdered, METH_VARARGS, "Returns all vertex data"},
	//{"export_gltf", (PyCFunction) flverExportGLTF, METH_NOARGS, "Returns all vertex data"},
	{NULL, NULL, 0, NULL}
};

struct PyModuleDef fromloaderModuleDef = {
	PyModuleDef_HEAD_INIT,
	"_fromloader",
	"flver module stuff i don't know",
	-1,
	fromloaderMethods,
	NULL,
	NULL,
	NULL,
	NULL
};

PyMODINIT_FUNC
PyInit__fromloader(void)
{
	PyObject *module;

	if (PyType_Ready(&flverType) < 0)
		return NULL;

	if (PyType_Ready(&mtdType) < 0)
		return NULL;
	
	if (PyType_Ready(&utilType) < 0)
		return NULL;
	
	/*if (PyType_Ready(&boneType) < 0)
		return NULL;*/
	
	module = PyModule_Create(&fromloaderModuleDef);

	/*Py_INCREF(&boneType);
	if (PyModule_AddObject(module, "Bone", (PyObject*) &boneType) < 0)
	{
		Py_DECREF(&boneType);
		Py_DECREF(module);
		return NULL;
	}*/

	Py_INCREF(&flverType);
	if (PyModule_AddObject(module, "flver2", (PyObject*) &flverType) < 0)
	{
		Py_DECREF(&flverType);
		Py_DECREF(module);
		return NULL;
	}

	Py_INCREF(&mtdType);
	if (PyModule_AddObject(module, "mtd", (PyObject*) &mtdType) < 0)
	{
		Py_DECREF(&mtdType);
		Py_DECREF(module);
		return NULL;
	}

	Py_INCREF(&utilType);
	if (PyModule_AddObject(module, "util", (PyObject*) &utilType) < 0)
	{
		Py_DECREF(&utilType);
		Py_DECREF(module);
		return NULL;
	}

	import_array();
	return module;
};
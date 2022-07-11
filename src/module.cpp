#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_14_API_VERSION
#include "module.h"

PyMODINIT_FUNC
PyInit_fromloader(void)
{
	PyObject *module;

	if (PyType_Ready(&flverType) < 0)
		return NULL;

	if (PyType_Ready(&mtdType) < 0)
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
	if (PyModule_AddObject(module, "Flver", (PyObject*) &flverType) < 0)
	{
		Py_DECREF(&flverType);
		Py_DECREF(module);
		return NULL;
	}

	Py_INCREF(&mtdType);
	if (PyModule_AddObject(module, "Mtd", (PyObject*) &mtdType) < 0)
	{
		Py_DECREF(&mtdType);
		Py_DECREF(module);
		return NULL;
	}

	import_array();
	return module;
};
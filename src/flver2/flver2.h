#pragma once
#include "../stdafx.h"

typedef struct
{
	PyObject_HEAD
	cfr::FLVER2* asset;
	PyObject* filePath;
	PyObject* materialList;
	//PyObject* mtdList;
	//PyObject* textureList;

	//bones are just gonna be a sucky stupid list of lists.
	//deal with it. i don't care anymore. there's no labels.
	PyObject* boneList;
	int boneCount;
	int meshCount;
} flverObject;

//members to expose to python
static PyMemberDef flverMembers[] = {
	{"filepath", T_STRING, offsetof(flverObject, filePath), READONLY, "file path"},
	//{"mtds", T_OBJECT, offsetof(flverObject, mtdList), READONLY, "mtd names"},
	//{"textures", T_OBJECT, offsetof(flverObject, textureList), READONLY, "tex list"},
	{"materials", T_OBJECT, offsetof(flverObject, materialList), READONLY, "mats"},
	{"bones", T_OBJECT, offsetof(flverObject, boneList), READONLY, "bones"},
	{"meshcount", T_INT, offsetof(flverObject, meshCount), READONLY, "meshcount"},
	{"bonecount", T_INT, offsetof(flverObject, boneCount), READONLY, "bonecount"},
	{NULL}
};

//internal
void flverDealloc (flverObject* self);

PyObject* flverNew(PyTypeObject* type, PyObject* args, PyObject* kwds);

int flverInit(flverObject* self, PyObject* args, PyObject* kwds);

//user facing stuff
PyObject* flverGetFaceset(flverObject* self, PyObject *args);

PyObject* flverGetVertData(flverObject* self, PyObject *args);

PyObject* flverGetVertDataOrdered(flverObject* self, PyObject *args);

PyObject* flverGenerateArmature(flverObject* self, PyObject * args);

static PyMethodDef flverMethods[] = {
	{"get_faceset", (PyCFunction) flverGetFaceset, METH_VARARGS, "Returns tris for faces"},
	{"get_vertex_data", (PyCFunction) flverGetVertData, METH_VARARGS, "Returns all vertex data"},
	//{"get_vertex_data_ordered", (PyCFunction) flverGetVertDataOrdered, METH_VARARGS, "Returns all vertex data"},
	//{"export_gltf", (PyCFunction) flverExportGLTF, METH_NOARGS, "Returns all vertex data"},
	{NULL, NULL, 0, NULL}
};

//boiler plate junk
static PyTypeObject flverType = {
	PyVarObject_HEAD_INIT(NULL, 0)
	"fromloader.Flver",							/*tp_name*/
	sizeof(flverObject),						/*tp_basicsize*/
	0,                                          /*tp_itemsize*/
	(destructor)flverDealloc,					/*tp_dealloc*/
	0,                                          /*tp_print*/
	0,                                          /*tp_getattr*/
	0,                                          /*tp_setattr*/
	0,                                          /*tp_compare*/
	0,                                          /*tp_repr*/
	0,                                          /*tp_as_number*/
	0,                                          /*tp_as_sequence*/
	0,                                          /*tp_as_mapping*/
	0,                                          /*tp_hash */
	0,                                          /*tp_call*/
	0,                                          /*tp_str*/
	0,                                          /*tp_getattro*/
	0,                                          /*tp_setattro*/
	0,                                          /*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /*tp_flags*/
	0,                                          /*tp_doc*/
	0,											/*tp_traverse*/
	0,											/*tp_clear*/
	0,                                          /*tp_richcompare*/
	0,                                          /*tp_weaklistoffset*/
	0,                                          /*tp_iter*/
	0,                                          /*tp_iternext*/
	flverMethods, 								/*tp_methods*/
	flverMembers,								/*tp_members*/
	0,                                          /*tp_getsets*/
	0,                                          /*tp_base*/
	0,                                          /*tp_dict*/
	0,                                          /*tp_descr_get*/
	0,                                          /*tp_descr_set*/
	0,                                          /*tp_dictoffset*/
	(initproc)flverInit,						/*tp_init*/
	0,                                          /*tp_alloc*/
	flverNew,									/*tp_new*/
};
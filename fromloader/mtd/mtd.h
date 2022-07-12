#pragma once
#include "../stdafx.h"

typedef struct
{
	PyObject_HEAD
	cfr::MTD* asset;
	PyObject* filePath;
	PyObject* params;
	int paramCount;
} mtdObject;

//members to expose to python
static PyMemberDef mtdMembers[] = {
	{"filepath", T_STRING, offsetof(mtdObject, filePath), READONLY, "file path"},
	{"params", T_OBJECT, offsetof(mtdObject, params), READONLY, "params"},
	{NULL}
};

void mtdDealloc (mtdObject* self);

PyObject* mtdNew(PyTypeObject* type, PyObject* args, PyObject* kwds);

int mtdInit(mtdObject* self, PyObject* args, PyObject* kwds);

static PyMethodDef mtdMethods[] = {
	{NULL, NULL, 0, NULL}
};

static PyTypeObject mtdType = {
	PyVarObject_HEAD_INIT(NULL, 0)
	"fromloader.MTD",							/*tp_name*/
	sizeof(mtdObject),							/*tp_basicsize*/
	0,                                          /*tp_itemsize*/
	(destructor)mtdDealloc,						/*tp_dealloc*/
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
	mtdMethods, 								/*tp_methods*/
	mtdMembers,									/*tp_members*/
	0,                                          /*tp_getsets*/
	0,                                          /*tp_base*/
	0,                                          /*tp_dict*/
	0,                                          /*tp_descr_get*/
	0,                                          /*tp_descr_set*/
	0,                                          /*tp_dictoffset*/
	(initproc)mtdInit,							/*tp_init*/
	0,                                          /*tp_alloc*/
	(newfunc)mtdNew,							/*tp_new*/
};
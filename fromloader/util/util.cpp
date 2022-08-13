#include "util.h"

PyObject* utilNew(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
	utilObject* self = (utilObject*)type->tp_alloc(type,0);

	if (self != NULL)
	{
		
	}

	return (PyObject*) self;
};

void utilDealloc(utilObject* self)
{
	Py_TYPE(self)->tp_free((PyObject*)self);
};

int utilInit(utilObject* self, PyObject* args, PyObject* kwds)
{
	return 0;
};

//pass in bmvertdeform object, bone indices, then bone weights
PyObject* turboStupid(utilObject* self, PyObject* args)
{
	PyObject* o;
	if(!PyArg_ParseTuple(args,"O",&o))
	{
		return NULL;
	}

	PyTypeObject* type = o->ob_type;
	//printf("Got type: %s\n",type->tp_name);

	BPy_BMDeformVert* dvert = (BPy_BMDeformVert*)o;
	/*printf("test: %i\n",dvert->data->dw[0].def_nr);
	printf("test: %i\n",dvert->data->dw[1].def_nr);
	printf("test: %i\n",dvert->data->dw[2].def_nr);
	printf("test: %i\n",dvert->data->dw[3].def_nr);*/

	//free(dvert->data->dw);
	//dvert->data->dw = (MDeformWeight*)malloc(sizeof(MDeformWeight) * 4);
	for(int i = 0; i < 4; i++)
	{
		dvert->data->dw[i].def_nr = i;
		dvert->data->dw[i].weight = (float)i/3;
	}

	PyObject* value = Py_BuildValue("i",1);
	return value;
};

PyObject* printCdl(utilObject* self, PyObject* args)
{
	PyObject* o;
	if(!PyArg_ParseTuple(args,"O",&o))
	{
		return NULL;
	}
	//#PyObject *o = NULL;
	//PyArg_ParseTuple(args, "O", &o);
	printf("got something!\n");
	PyTypeObject* type = o->ob_type;
	printf("Got type: %s\n",type->tp_name);

	printf("bonecount: %i\n",((flverObject3*)o)->boneCount);
	((flverObject3*)o)->boneCount = 10;

	PyObject* value = Py_BuildValue("i",1);

	return value;
};
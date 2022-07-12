#include "mtd.h"

PyObject* mtdNew(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
	mtdObject* self = (mtdObject*)type->tp_alloc(type,0);

	if (self != NULL)
	{
		self->asset = NULL;
		self->filePath = NULL;
		self->params = NULL;
		self->paramCount = 0;
	}

	return (PyObject*) self;
};

void mtdDealloc (mtdObject* self)
{
	Py_XDECREF(self->filePath);
	Py_XDECREF(self->params);
	Py_TYPE(self)->tp_free((PyObject*)self);
};

int mtdInit(mtdObject* self, PyObject* args, PyObject* kwds)
{
	static char* kwlist[] = {"filepath",NULL};
	const char* c_filePath = NULL;

	if(!PyArg_ParseTupleAndKeywords(args,kwds,"s:__init__",kwlist,&c_filePath))
	{
		return -1;
	}

	if(c_filePath)
	{
		import_array();

		PyObject* filePath = PyUnicode_FromString(c_filePath);
		Py_INCREF(filePath);
		self->filePath = filePath;
		Py_INCREF(self->filePath);

		printf("Got mtd path: %s\n",PyUnicode_AsUTF8(filePath));

		self->asset = new cfr::MTD(PyUnicode_AsUTF8(filePath));
		/*try
		{
			self->asset = new cfr::MTD(PyUnicode_AsUTF8(filePath));
		}
		catch(...)
		{
			printf("error occured!\n");
		}*/
		
		printf("managed to open mtd!\n");
		self->paramCount = self->asset->mtdData.lists.paramCount;
		printf("param count = %i\n",self->paramCount);

		PyObject* paramList = PyList_New(self->paramCount);

		for(int i = 0; i < self->paramCount; i++)
		{
			PyObject* param = PyList_New(3);
			cfr::MTD::Param* p = &self->asset->mtdData.lists.params[i];

			PyObject* nameStr = _PyUnicode_FromASCII(p->name.str,p->name.length);
			PyList_SetItem(param,0,nameStr);
			PyObject* typeStr = _PyUnicode_FromASCII(p->type.str,p->type.length);
			PyList_SetItem(param,1,typeStr);

			PyObject* value;
			float* fv = p->value.floatValues;

			if(strncmp(p->type.str,"Bool",4) == 0)
			{
				value = Py_BuildValue("p",(bool)&p->value.byteValues[0]);
			}
			else if(strncmp(p->type.str,"Int",3) == 0)
			{
				value = Py_BuildValue("i",(bool)&p->value.intValues[0]);
			}
			else if(strncmp(p->type.str,"Int2",4) == 0)
			{
				value = Py_BuildValue("[ii]",&p->value.intValues[0],&p->value.intValues[1]);
			}
			else if(strncmp(p->type.str,"Float",5) == 0)
			{
				value = Py_BuildValue("f",&fv[0]);
			}
			else if(strncmp(p->type.str,"Float2",6) == 0)
			{
				value = Py_BuildValue("[ff]",&fv[0],&fv[1]);
			}
			else if(strncmp(p->type.str,"Float3",6) == 0)
			{
				value = Py_BuildValue("[fff]",&fv[0],&fv[1],&fv[2]);
			}
			else if(strncmp(p->type.str,"Float4",6) == 0)
			{
				value = Py_BuildValue("[ffff]",&fv[0],&fv[1],&fv[2],&fv[3]);
			}
			PyList_SetItem(param,2,value);
		}

		Py_IncRef(paramList);
		self->params = paramList;
		Py_IncRef(self->params);
	};

	return 0;
};
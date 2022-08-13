#include "mtd.h"
#include <map>

PyObject* mtdNew(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
	mtdObject* self = (mtdObject*)type->tp_alloc(type,0);

	if (self != NULL)
	{
		self->params = NULL;
	}

	return (PyObject*) self;
};

void mtdDealloc (mtdObject* self)
{
	Py_XDECREF(self->params);
	Py_TYPE(self)->tp_free((PyObject*)self);
};

int mtdInit(mtdObject* self, PyObject* args, PyObject* kwds)
{
	static char* kwlist[] = {"filepath",NULL};
	const char* c_filePath = NULL;

	if(!PyArg_ParseTuple(args,"s",&c_filePath))
	{
		return NULL;
	}

	if(c_filePath)
	{
		//import_array();
		//printf("[C]\tAttemping to load MTD: %s...\n",c_filePath);

		cfr::MTD* asset = new cfr::MTD(c_filePath);

		//self->paramCount = self->asset->mtdData.lists.paramCount;
		//printf("[C] MTD param count: %i\n",asset->mtdData->lists->paramCount);

		self->params = PyList_New(asset->mtdData->lists->paramCount);

		for(int i = 0; i < asset->mtdData->lists->paramCount; i++)
		{
			PyObject* param = PyList_New(3);
			cfr::MTD::Param* p = asset->mtdData->lists->params[i];

			int len = 0; setlocale(LC_ALL, "");

			
			int nameLen = cfr::jisToUtf8(p->name->str,p->name->length,NULL);
			char* name = (char*)malloc(nameLen);
			cfr::jisToUtf8(p->name->str,p->name->length,name);

			PyObject* puname = Py_BuildValue("s#",name,nameLen);
			PyList_SetItem(param,0,puname);

			int typeLen = cfr::jisToUtf8(p->type->str,p->type->length,NULL);
			char* type = (char*)malloc(typeLen);
			cfr::jisToUtf8(p->type->str,p->type->length,type);
			//printf("type: %s\n",type);

			PyObject* putype = Py_BuildValue("s#",type,typeLen);
			PyList_SetItem(param,1,putype);

			PyObject* value;
			float* fv = p->value->floatValues;

			if(strncmp(type,"Bool",4) == 0)
			{
				value = Py_BuildValue("b",(bool)p->value->byteValues[0]);
			}
			else if(strncmp(type,"bool",4) == 0)
			{
				value = Py_BuildValue("b",(bool)p->value->byteValues[0]);
			}
			else if(strncmp(type,"Int",3) == 0)
			{
				value = Py_BuildValue("i",p->value->intValues[0]);
			}
			else if(strncmp(type,"int",3) == 0)
			{
				value = Py_BuildValue("i",p->value->intValues[0]);
			}
			else if(strncmp(type,"Int2",4) == 0)
			{
				value = Py_BuildValue("[ii]",p->value->intValues[0],p->value->intValues[1]);
			}
			else if(strncmp(type,"Float",5) == 0)
			{
				value = Py_BuildValue("f",fv[0]);
			}
			else if(strncmp(type,"float",5) == 0)
			{
				value = Py_BuildValue("f",fv[0]);
			}
			else if(strncmp(type,"Float2",6) == 0)
			{
				value = Py_BuildValue("[ff]",fv[0],fv[1]);
			}
			else if(strncmp(type,"Float3",6) == 0)
			{
				value = Py_BuildValue("[fff]",fv[0],fv[1],fv[2]);
			}
			else if(strncmp(type,"Float4",6) == 0)
			{
				value = Py_BuildValue("[ffff]",fv[0],fv[1],fv[2],fv[3]);
			}
			else
			{
				printf("Got unknown type: '%s'\n",type);
				throw std::runtime_error("AAAAAAH!\n");
			}
			PyList_SetItem(param,2,value);

			PyList_SetItem(self->params,i,param);

			free(name);
			name = NULL;

			free(type);
			type = NULL;
		}

		//printf("KILL\n");
		asset->~MTD();
	};

	return 0;
};
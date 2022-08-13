#pragma once
#include "../stdafx.h"

typedef struct CustomDataLayer {
	/** Type of data in layer. */
	int type;
	/** In editmode, offset of layer in block. */
	int offset;
	/** General purpose flag. */
	int flag;
	/** Number of the active layer of this type. */
	int active;
	/** Number of the layer to render. */
	int active_rnd;
	/** Number of the layer to render. */
	int active_clone;
	/** Number of the layer to render. */
	int active_mask;
	/** Shape keyblock unique id reference. */
	int uid;
	/** Layer name, MAX_CUSTOMDATA_LAYER_NAME. */
	char name[64];
	/** Layer data. */
	void *data;
} CustomDataLayer;

typedef struct CustomDataExternal {
	/** FILE_MAX. */
	char filename[1024];
} CustomDataExternal;

typedef struct CustomData {
	/** CustomDataLayers, ordered by type. */
	CustomDataLayer *layers;
	/**
	 * runtime only! - maps types to indices of first layer of that type,
	 * MUST be >= CD_NUMTYPES, but we cant use a define here.
	 * Correct size is ensured in CustomData_update_typemap assert().
	 */
	int typemap[50];
	char _pad[4];
	/** Number of layers, size of layers array. */
	int totlayer, maxlayer;
	/** In editmode, total size of all data layers. */
	int totsize;
	/** (BMesh Only): Memory pool for allocation of blocks. */
	struct BLI_mempool *pool;
	/** External file storing customdata layers. */
	CustomDataExternal *external;
} CustomData;

typedef struct
{
	PyObject_HEAD
	cfr::FLVER2* asset;
	PyObject* filePath;
	PyObject* materialList;
	PyObject* meshList;
	//PyObject* mtdList;
	//PyObject* textureList;
	//bones are just gonna be a sucky stupid list of lists.
	//deal with it. i don't care anymore. there's no labels.
	PyObject* boneList;
	PyObject* dummyList;
	PyObject* facesetList;
	PyObject* vertexBufferList;
	PyObject* normalFacesList;

	int boneCount;
	int meshCount;
	int materialCount;
	int dummyCount;
} flverObject3;

typedef struct MDeformWeight {
	/** The index for the vertex group, must *always* be unique when in an array. */
	unsigned int def_nr;
	/** Weight between 0.0 and 1.0. */
	float weight;
} MDeformWeight;

typedef struct MDeformVert {
	struct MDeformWeight *dw;
	int totweight;
	/** Flag is only in use as a run-time tag at the moment. */
	int flag;
} MDeformVert;

typedef struct BPy_BMDeformVert {
	PyObject_VAR_HEAD MDeformVert *data;
} BPy_BMDeformVert;

typedef struct
{
	PyObject_HEAD
} utilObject;

static PyMemberDef utilMembers[] = {
	{NULL}
};

typedef struct BPy_BMVert {
	PyObject_VAR_HEAD struct BMesh *bm; /* keep first */
	struct BMVert *v;
} BPy_BMVert;

typedef struct ListBase {
	void *first, *last;
} ListBase;

typedef struct BMesh {
	int totvert, totedge, totloop, totface;
	int totvertsel, totedgesel, totfacesel;

	/* flag index arrays as being dirty so we can check if they are clean and
	* avoid looping over the entire vert/edge/face/loop array in those cases.
	* valid flags are - BM_VERT | BM_EDGE | BM_FACE | BM_LOOP. */
	char elem_index_dirty;

	/* flag array table as being dirty so we know when its safe to use it,
	* or when it needs to be re-created */
	char elem_table_dirty;

	/* element pools */
	struct BLI_mempool *vpool, *epool, *lpool, *fpool;

	/* mempool lookup tables (optional)
	* index tables, to map indices to elements via
	* BM_mesh_elem_table_ensure and associated functions.  don't
	* touch this or read it directly.\
	* Use BM_mesh_elem_table_ensure(), BM_vert/edge/face_at_index() */
	BMVert **vtable;
	//BMEdge **etable;
	void **etable;
	//BMFace **ftable;
	void **ftable;

	/* size of allocated tables */
	int vtable_tot;
	int etable_tot;
	int ftable_tot;

	/* operator api stuff (must be all NULL or all alloc'd) */
	struct BLI_mempool *vtoolflagpool, *etoolflagpool, *ftoolflagpool;

	uint32_t use_toolflags : 1;

	int toolflag_index;
	struct BMOperator *currentop;

	CustomData vdata, edata, ldata, pdata;

	#ifdef USE_BMESH_HOLES
	struct BLI_mempool *looplistpool;
	#endif

	struct MLoopNorSpaceArray *lnor_spacearr;
	char spacearr_dirty;

	/* should be copy of scene select mode */
	/* stored in BMEditMesh too, this is a bit confusing,
	* make sure they're in sync!
	* Only use when the edit mesh cant be accessed - campbell */
	short selectmode;

	/* ID of the shape key this bmesh came from */
	int shapenr;

	int totflags;
	ListBase selected;

	//BMFace *act_face;
	void *act_face;

	ListBase errorstack;

	void *py_handle;
} BMesh;

void utilDealloc(utilObject* self);

PyObject* utilNew(PyTypeObject* type, PyObject* args, PyObject* kwds);

int utilInit(utilObject* self, PyObject* args, PyObject* kwds);

PyObject* printCdl(utilObject* self, PyObject* args);

PyObject* turboStupid(utilObject* self, PyObject* args);

static PyMethodDef utilMethods[] = {
	{"print_custom_data_layer", (PyCFunction) printCdl, METH_VARARGS, "print cdl stuff to console"},
	{"turbo_stupid", (PyCFunction) turboStupid, METH_VARARGS, "print cdl stuff to console"},
	{NULL, NULL, 0, NULL}
};

static PyTypeObject utilType = {
	PyVarObject_HEAD_INIT(NULL, 0)
	"fromloader.util",							/*tp_name*/
	sizeof(utilObject),							/*tp_basicsize*/
	0,                                          /*tp_itemsize*/
	(destructor)utilDealloc,					/*tp_dealloc*/
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
	utilMethods, 								/*tp_methods*/
	utilMembers,								/*tp_members*/
	0,                                          /*tp_getsets*/
	0,                                          /*tp_base*/
	0,                                          /*tp_dict*/
	0,                                          /*tp_descr_get*/
	0,                                          /*tp_descr_set*/
	0,                                          /*tp_dictoffset*/
	(initproc)utilInit,							/*tp_init*/
	0,                                          /*tp_alloc*/
	utilNew,									/*tp_new*/
};
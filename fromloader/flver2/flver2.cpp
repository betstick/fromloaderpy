#include "flver2.h"

PyObject* flverNew(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
	flverObject* self = (flverObject*)type->tp_alloc(type,0);

	if (self != NULL)
	{
		self->filePath = NULL;
		self->asset = NULL;

		//self->mtdList = NULL;
		self->materialList = NULL;
		self->meshCount = 0;
	}

	return (PyObject*) self;
};

void flverDealloc (flverObject* self)
{
	Py_XDECREF(self->filePath);
	Py_XDECREF(self->materialList);
	//Py_XDECREF(self->mtdList);
	Py_TYPE(self)->tp_free((PyObject*)self);
};

/*struct FixedBone
{
	wchar_t* name;
	int nameLength;
	float headPos[3];
	float tailPos[3];
	bool initialized;
};*/

/*void fixBone()
{

};

std::vector<FixedBone> fixBones(cfr::FLVER2::Bone* boneArr, int boneCount)
{
	
};*/

int flverInit(flverObject* self, PyObject* args, PyObject* kwds)
{
	static char* kwlist[2] = {"filepath",NULL};

	const char* c_filePath = NULL;

	if(!PyArg_ParseTupleAndKeywords(args,kwds,"s:__init__",kwlist,&c_filePath))
	{
		return -1;
	}

	if(c_filePath)
	{
		PyObject* filePath = PyUnicode_FromString(c_filePath);
		Py_INCREF(filePath);
		self->filePath = filePath;
		Py_INCREF(self->filePath);

		//printf("Got path: %s\n",PyUnicode_AsUTF8(filePath));
		
		self->asset = new cfr::FLVER2(PyUnicode_AsUTF8(filePath));
		
		self->meshCount = self->asset->header.meshCount;

		//init mtd list
		/*PyObject* mtdList = PyList_New(self->asset->header.materialCount);
		
		for(int i = 0; i < self->asset->header.materialCount; i++)
		{
			wchar_t* wchstr = self->asset->materials[i].mtdName;
			int len = self->asset->materials[i].mtdNameLength;

			PyObject* mtdName = PyUnicode_FromWideChar(wchstr,len);
			PyList_SetItem(mtdList,i,mtdName);
			//Py_IncRef(mtdName);
		}

		Py_IncRef(mtdList);
		self->mtdList = mtdList;
		Py_IncRef(self->mtdList);*/

		//init bone list
		PyObject* boneList = PyList_New(self->asset->header.boneCount);
		//printf("got bone count: %i\n",self->asset->header.boneCount);
		self->boneCount = self->asset->header.boneCount;

		//so there's no obvious or simple way i can find to init the boneObject
		//type i made in this module. not without making *another* module. or
		//doing some abstract obtuse boilerplate that makes no sense.
		//point is, python extension development sucks. python sucks.
		//the whole thing sucks. never touch python. its a terrible language
		//with terrible docs on arcane, obfuscated nonsense.
		//so instead we'll just use a big list of lists!
		//will it be slow? i don't care. armature generation costs no time
		//for maps cause they have one bone.
		//as for characters, they're already fast enough at importing.

		for(int i = 0; i < self->asset->header.boneCount; i++)
		{
			PyObject* bonearr = PyList_New(10);

			cfr::FLVER2::Bone* bone = &self->asset->bones[i];

			//name conversion
			wchar_t* name = bone->name;
			int length = bone->nameLength;
			PyObject* boneName = PyUnicode_FromWideChar(name,length);
			PyList_SetItem(bonearr,0,boneName);

			//translation conversion
			cfr::cfr_vec3 t = bone->translation;
			PyObject* translation = Py_BuildValue("[fff]",t.x,t.y,t.z);
			PyList_SetItem(bonearr,1,translation);

			//rotation conversion
			cfr::cfr_vec3 r = bone->rot;
			PyObject* rotation = Py_BuildValue("[fff]",r.x,r.y,r.z);
			PyList_SetItem(bonearr,2,rotation);

			//scale conversion
			cfr::cfr_vec3 s = bone->scale;
			PyObject* scale = Py_BuildValue("[fff]",s.x,s.y,s.z);
			PyList_SetItem(bonearr,3,scale);

			//bounding box min conversion
			cfr::cfr_vec3 bbmi = bone->boundingBoxMin;
			PyObject* bbMin = Py_BuildValue("[fff]",bbmi.x,bbmi.y,bbmi.z);
			PyList_SetItem(bonearr,4,bbMin);

			//bounding box max conversion
			cfr::cfr_vec3 bbma = bone->boundingBoxMax;
			PyObject* bbMax = Py_BuildValue("[fff]",bbma.x,bbma.y,bbma.z);
			PyList_SetItem(bonearr,5,bbMax);

			PyObject* pi = Py_BuildValue("i",bone->parentIndex);
			PyList_SetItem(bonearr,6,pi);

			PyObject* ci = Py_BuildValue("i",bone->childIndex);
			PyList_SetItem(bonearr,7,ci);

			PyObject* nsi = Py_BuildValue("i",bone->nextSiblingIndex);
			PyList_SetItem(bonearr,8,nsi);

			PyObject* psi = Py_BuildValue("i",bone->previousSiblingIndex);
			PyList_SetItem(bonearr,9,psi);
					
			PyList_SetItem(boneList,i,bonearr);
		}

		Py_IncRef(boneList);
		self->boneList = boneList;
		Py_IncRef(self->boneList);

		//setup the material stuff here wooooo and textures! and mtdparams!!!
		PyObject* materialList = PyList_New(self->asset->header.materialCount);

		for(int i = 0; i < self->asset->header.materialCount; i++)
		{
			PyObject* mat = PyList_New(6);

			PyObject* name = PyUnicode_FromWideChar(self->asset->materials[i].name,self->asset->materials[i].nameLength);
			PyList_SetItem(mat,0,name);

			PyObject* mtd = PyUnicode_FromWideChar(self->asset->materials[i].mtdName,self->asset->materials[i].mtdNameLength);
			PyList_SetItem(mat,1,mtd);

			PyObject* flags = Py_BuildValue("i",self->asset->materials[i].header.flags);
			PyList_SetItem(mat,2,flags);

			PyObject* gxIndex = Py_BuildValue("i",self->asset->materials[i].gxIndex);
			PyList_SetItem(mat,3,gxIndex);
			
			PyObject* texList = PyList_New(self->asset->materials[i].header.textureCount);
			for(int t = 0; t < self->asset->materials[i].header.textureCount; t++)
			{
				int texindex = self->asset->materials[i].header.textureIndex + t;

				PyObject* tex = PyList_New(4);

				PyObject* path = PyUnicode_FromWideChar(self->asset->textures[texindex].path,self->asset->textures[texindex].pathLength);
				PyList_SetItem(tex,0,path);

				PyObject* x = Py_BuildValue("f",self->asset->textures[texindex].scale.x);
				PyList_SetItem(tex,1,x);

				PyObject* y = Py_BuildValue("f",self->asset->textures[texindex].scale.y);
				PyList_SetItem(tex,2,y);

				PyObject* textype = PyUnicode_FromWideChar(self->asset->textures[texindex].type,self->asset->textures[texindex].typeLength);
				PyList_SetItem(tex,0,textype);

				PyList_SetItem(texList,t,tex);
			}
			PyList_SetItem(mat,4,texList);

			PyList_SetItem(materialList,i,mat);
		}

		Py_IncRef(materialList);
		self->materialList = materialList;
		Py_IncRef(self->materialList);

		//printf("managed to open flver! meshcount: %i\n",self->meshCount);
	};

	return 0;
};

PyObject* flverGenerateArmature(flverObject* self, PyObject * args)
{
	return nullptr;
};

PyObject* flverGetFaceset(flverObject* self, PyObject *args)
{
	static char* kwlist[] = {"mesh","faceset",NULL};
	
	int mesh_i;
	int faceset_i; //does nothing

	if(!PyArg_ParseTuple(args,"ii",&mesh_i,&faceset_i))
	{
		return NULL;
	}

	cfr::FLVER2::Faceset* facesetp = nullptr;
	cfr::FLVER2::Mesh meshp = self->asset->meshes[mesh_i];

	if(mesh_i > self->asset->header.meshCount - 1)
	{
		printf("INVALID MESH INDEX!\n");
		return NULL;
	}
	
	uint64_t lowest_flags = LLONG_MAX;
	
	for(int mfsi = 0; mfsi < meshp.header.facesetCount; mfsi++)
	{
		int fsindex = meshp.facesetIndices[mfsi];
		if(self->asset->facesets[fsindex].header.flags < lowest_flags)
		{
			facesetp = &self->asset->facesets[fsindex];
			lowest_flags = facesetp->header.flags;
		}
	}

	if(facesetp == nullptr)
	{
		printf("mesh_index:%i\n",mesh_i);
		printf("ARGH WE BE AT A NULL POINTER! ARGAGHAHGHAHG!\n");
	}

	facesetp->triangulate();
	//printf("triangulated %i tris\n",faceset->triCount);

	npy_intp dim[2];
	dim[0] = facesetp->triCount / 3;
	dim[1] = 3;

	PyObject* nparr = PyArray_SimpleNewFromData(
		2,dim,NPY_INT32,&facesetp->triList[0]
	);
	Py_IncRef(nparr);

	return nparr;
};

//returns list of nparrys
PyObject* flverGetVertData(flverObject* self, PyObject *args)
{
	import_array(); //THIS SHOULD NOT HAVE TO BE HERE BUT IT DOES.
	int mesh_index;

	if(!PyArg_ParseTuple(args,"i",&mesh_index))
	{
		return NULL;
	}

	//printf("mesh_index: %i\n",mesh_index);

	cfr::FLVER2::Mesh* mesh = &self->asset->meshes[mesh_index];

	int vertCount = 0;

	for(int vbi = 0; vbi < self->asset->meshes[mesh_index].header.vertexBufferCount; vbi++) //vertex buffer index
	{
		int vb_index = self->asset->meshes[mesh_index].vertexBufferIndices[vbi];
		vertCount += self->asset->vertexBuffers[vb_index].header.vertexCount;
	}

	//printf("vertcount = %i\n",vertCount);

	int uvCount = 0;
	int colorCount = 0;
	int tanCount = 0;

	int uvFactor = 1024;
	if(self->asset->header.version >= 0x2000F)
		uvFactor = 2048;
	
	//printf("uvFactor: %i\n",uvFactor);

	self->asset->getVertexData(mesh_index,&uvCount,&colorCount,&tanCount);
	
	//printf("colorCount: %i\n",colorCount);
	//printf("module vertcount: %u\n",vertCount);

	npy_intp dim1[1] = {vertCount};
	npy_intp dim3[2] = {vertCount,3};
	npy_intp dim4[2] = {vertCount,4};
	npy_intp tanDim[3] = {vertCount,tanCount,3};
	npy_intp uvDim[3] = {vertCount,uvCount,2};
	npy_intp colorDim[3] = {vertCount,colorCount,4};

	//printf("colrdim: %i\n",colorDim[0]);

	PyObject* npPositions = PyArray_SimpleNewFromData(2,dim3,NPY_FLOAT32,mesh->vertexData->positions);
	PyObject* npNoneWeights = PyArray_SimpleNewFromData(2,dim4,NPY_FLOAT32,&mesh->vertexData->bone_weights[0]);
	PyObject* npBoneIndices = PyArray_SimpleNewFromData(2,dim4,NPY_INT16,&mesh->vertexData->bone_indices[0]);
	PyObject* npNormals = PyArray_SimpleNewFromData(2,dim3,NPY_FLOAT32,&mesh->vertexData->normals[0]);
	PyObject* npNormalws = PyArray_SimpleNewFromData(1,dim1,NPY_INT32,&mesh->vertexData->normalws[0]);
	PyObject* npTangents = PyArray_SimpleNewFromData(3,tanDim,NPY_FLOAT32,&mesh->vertexData->tangents[0]);
	PyObject* npBitangets = PyArray_SimpleNewFromData(2,dim4,NPY_FLOAT32,&mesh->vertexData->bitangents[0]);
	PyObject* npUVs = PyArray_SimpleNewFromData(3,uvDim,NPY_FLOAT32,&mesh->vertexData->uvs[0]);
	PyObject* npColors = PyArray_SimpleNewFromData(3,colorDim,NPY_FLOAT32,&mesh->vertexData->colors[0]);

	//not sure if this is needed
	Py_IncRef(npPositions);
	Py_IncRef(npNoneWeights);
	Py_IncRef(npBoneIndices);
	Py_IncRef(npNormals);
	Py_IncRef(npNormalws);
	Py_IncRef(npTangents);
	Py_IncRef(npBitangets);
	Py_IncRef(npUVs);
	Py_IncRef(npColors);

	PyObject* list = PyList_New(9);
	Py_IncRef(list);
	PyList_SetItem(list,0,npPositions);
	PyList_SetItem(list,1,npNoneWeights);
	PyList_SetItem(list,2,npBoneIndices);
	PyList_SetItem(list,3,npNormals);
	PyList_SetItem(list,4,npNormalws);
	PyList_SetItem(list,5,npTangents);
	PyList_SetItem(list,6,npBitangets);
	PyList_SetItem(list,7,npUVs);
	PyList_SetItem(list,8,npColors);

	Py_IncRef(list);
	//self->meshData[mesh_index] = *list;

	//printf("list address: 0x%x\n",list);

	return list;
};

//more data, but might be faster since less jumping around?
PyObject* flverGetVertDataOrdered(flverObject* self, PyObject *args)
{
	int mesh_index;

	if(!PyArg_ParseTuple(args,"i",&mesh_index))
	{
		return NULL;
	}

	//printf("mesh_index: %i\n",mesh_index);

	cfr::FLVER2::Mesh* mesh = &self->asset->meshes[mesh_index];

	int vertCount = 0;

	for(int vbi = 0; vbi < self->asset->meshes[mesh_index].header.vertexBufferCount; vbi++) //vertex buffer index
	{
		int vb_index = self->asset->meshes[mesh_index].vertexBufferIndices[vbi];
		vertCount += self->asset->vertexBuffers[vb_index].header.vertexCount;
	}

	//printf("vertcount = %i\n",vertCount);

	int uvCount = 0;
	int colorCount = 0;
	int tanCount = 0;

	self->asset->getVertexData(mesh_index,&uvCount,&colorCount,&tanCount);

	int fsi = 0;
	uint64_t lowest_flags = LLONG_MAX;

	//printf("2uvcount:%i\n",uvCount);

	for(int mfsi = 0; mfsi < self->asset->meshes[mesh_index].header.facesetCount; mfsi++)
	{
		int fsindex = self->asset->meshes[mesh_index].facesetIndices[mfsi];
		if(self->asset->facesets[fsindex].header.flags < lowest_flags)
		{
			fsi = fsindex;
			lowest_flags = self->asset->facesets[fsindex].header.flags;
		}
	}

	self->asset->facesets[fsi].triangulate();
	//printf("fsi:%i\n",fsi);
	//printf("fsic:%i\n",self->asset->facesets[fsi].header.vertexIndexCount);

	int triCount = self->asset->facesets[fsi].triCount;
	//printf("tricount:%i\n",triCount);

	self->asset->getVertexDataOrdered(mesh_index,uvCount,colorCount,tanCount,vertCount);
	
	npy_intp dim1[1] = {triCount};
	npy_intp dim3[3] = {triCount,3,3};
	npy_intp dim4[2] = {triCount,4};
	npy_intp tanDim[3] = {triCount,tanCount,3};
	npy_intp uvDim[3] = {triCount,uvCount,2};
	npy_intp colorDim[3] = {triCount,colorCount,4};

	//printf("colrdim: %i\n",colorDim[0]);

	PyObject* npPositions = PyArray_SimpleNewFromData(3,dim3,NPY_FLOAT32,&mesh->vertexDataOrdered->positions[0]);
	PyObject* npNoneWeights = PyArray_SimpleNewFromData(2,dim4,NPY_FLOAT32,&mesh->vertexDataOrdered->bone_weights[0]);
	PyObject* npBoneIndices = PyArray_SimpleNewFromData(2,dim4,NPY_INT16,&mesh->vertexDataOrdered->bone_indices[0]);
	PyObject* npNormals = PyArray_SimpleNewFromData(2,dim3,NPY_FLOAT32,&mesh->vertexDataOrdered->normals[0]);
	PyObject* npNormalws = PyArray_SimpleNewFromData(1,dim1,NPY_INT32,&mesh->vertexDataOrdered->normalws[0]);
	PyObject* npTangents = PyArray_SimpleNewFromData(3,tanDim,NPY_FLOAT32,&mesh->vertexDataOrdered->tangents[0]);
	PyObject* npBitangets = PyArray_SimpleNewFromData(2,dim4,NPY_FLOAT32,&mesh->vertexDataOrdered->bitangents[0]);
	PyObject* npUVs = PyArray_SimpleNewFromData(3,uvDim,NPY_FLOAT32,&mesh->vertexDataOrdered->uvs[0]);
	PyObject* npColors = PyArray_SimpleNewFromData(3,colorDim,NPY_FLOAT32,&mesh->vertexDataOrdered->colors[0]);
	
	//not sure if this is needed
	Py_IncRef(npPositions);
	Py_IncRef(npNoneWeights);
	Py_IncRef(npBoneIndices);
	Py_IncRef(npNormals);
	Py_IncRef(npNormalws);
	Py_IncRef(npTangents);
	Py_IncRef(npBitangets);
	Py_IncRef(npUVs);
	Py_IncRef(npColors);

	PyObject* list = PyList_New(9);
	PyList_SetItem(list,0,npPositions);
	PyList_SetItem(list,1,npNoneWeights);
	PyList_SetItem(list,2,npBoneIndices);
	PyList_SetItem(list,3,npNormals);
	PyList_SetItem(list,4,npNormalws);
	PyList_SetItem(list,5,npTangents);
	PyList_SetItem(list,6,npBitangets);
	PyList_SetItem(list,7,npUVs);
	PyList_SetItem(list,8,npColors);

	Py_IncRef(list);
	//self->meshData[mesh_index] = *list;

	return list;
};

/*static PyObject* flverExportGLTF(flverObject* self, PyObject *args)
{
	cfr::exportGLTF(self->asset);

	return Py_BuildValue("i",0);
};*/
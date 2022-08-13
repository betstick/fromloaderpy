#include "flver2.h"

PyObject* flverNew(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
	flverObject* self = (flverObject*)type->tp_alloc(type,0);

	if (self != NULL)
	{
		self->filePath = NULL;
		self->asset = NULL;

		self->materialList = NULL;
		self->meshList = NULL;
		self->boneList = NULL;
		self->dummyCount = NULL;
		self->facesetList = NULL;
		self->vertexBufferList = NULL;
		self->normalFacesList = NULL;

		self->materialCount = 0;
		self->meshCount = 0;
		self->boneCount = 0;
		self->dummyCount = 0;
	}

	return (PyObject*) self;
};

void flverDealloc(flverObject* self)
{
	for(int i = 0; i < self->meshCount; i++)
	{
		cfr::FLVER2::Mesh* mesh = self->asset->meshes[i];
		cfr::FLVER2::Faceset* facesetp = NULL;
		
		uint64_t lowest_flags = LLONG_MAX;
		for(int mfsi = 0; mfsi < mesh->header.facesetCount; mfsi++)
		{
			int fsindex = mesh->facesetIndices[mfsi];
			if(self->asset->facesets[fsindex]->header.flags < lowest_flags)
			{
				facesetp = self->asset->facesets[fsindex];
				lowest_flags = facesetp->header.flags;
			}
		}

		free(facesetp->triList);
		facesetp->triList = NULL;

		free(facesetp->facesetNorms);
		facesetp->facesetNorms = NULL;

		free(mesh->vertexData->positions);
		mesh->vertexData->positions = NULL;

		free(mesh->vertexData->bone_weights);
		mesh->vertexData->bone_weights = NULL;

		free(mesh->vertexData->bone_indices);
		mesh->vertexData->bone_indices = NULL;

		free(mesh->vertexData->normals);
		mesh->vertexData->normals = NULL;

		free(mesh->vertexData->normalws);
		mesh->vertexData->normalws = NULL;

		free(mesh->vertexData->tangents);
		mesh->vertexData->tangents = NULL;

		free(mesh->vertexData->bitangents);
		mesh->vertexData->bitangents = NULL;

		free(mesh->vertexData->uvs);
		mesh->vertexData->uvs = NULL;

		free(mesh->vertexData->colors);
		mesh->vertexData->colors = NULL;

		free(mesh->vertexData);
		mesh->vertexData = NULL;
	}
	//free(facesetp->triList);
	//facesetp->triList = NULL;

	delete(self->asset);
	Py_XDECREF(self->filePath);
	Py_XDECREF(self->materialList);
	Py_XDECREF(self->meshList);
	Py_XDECREF(self->boneList);
	Py_XDECREF(self->dummyList);
	Py_XDECREF(self->facesetList);
	Py_XDECREF(self->vertexBufferList);
	Py_XDECREF(self->normalFacesList);
	//Py_XDECREF(self->mtdList);
	Py_TYPE(self)->tp_free((PyObject*)self);
};

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
		import_array(); //THIS SHOULD NOT HAVE TO BE HERE BUT IT DOES.
		self->filePath = PyUnicode_FromString(c_filePath);
		//Py_INCREF(self->filePath);

		//printf("[C] Got path: %s\n",PyUnicode_AsUTF8(self->filePath));
		
		#define VALIDATE_ALL //for error messages in flver loading
		self->asset = new cfr::FLVER2(c_filePath);
		
		self->materialCount = self->asset->header.materialCount;
		self->meshCount = self->asset->header.meshCount;
		self->boneCount = self->asset->header.boneCount;
		self->dummyCount = self->asset->header.dummyCount;
		
		//printf("got bone count: %i\n",self->asset->header.boneCount);

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

		self->boneList = PyList_New(self->asset->header.boneCount);

		for(int i = 0; i < self->asset->header.boneCount; i++)
		{
			PyObject* bonearr = PyList_New(10);

			cfr::FLVER2::Bone* bone = self->asset->bones[i];

			//name conversion
			PyList_SetItem(bonearr,0,PyUnicode_FromWideChar(bone->name,bone->nameLength));

			//translation conversion
			cfr::cfr_vec3 t = bone->translation;
			PyList_SetItem(bonearr,1,Py_BuildValue("[fff]",t.x,t.y,t.z));

			//rotation conversion
			cfr::cfr_vec3 r = bone->rot;
			PyList_SetItem(bonearr,2,Py_BuildValue("[fff]",r.x,r.y,r.z));

			//scale conversion
			cfr::cfr_vec3 s = bone->scale;
			PyList_SetItem(bonearr,3,Py_BuildValue("[fff]",s.x,s.y,s.z));

			//bounding box min conversion
			cfr::cfr_vec3 bbmi = bone->boundingBoxMin;
			PyList_SetItem(bonearr,4,Py_BuildValue("[fff]",bbmi.x,bbmi.y,bbmi.z));

			//bounding box max conversion
			cfr::cfr_vec3 bbma = bone->boundingBoxMax;
			PyList_SetItem(bonearr,5,Py_BuildValue("[fff]",bbma.x,bbma.y,bbma.z));

			PyList_SetItem(bonearr,6,Py_BuildValue("i",bone->parentIndex));

			PyList_SetItem(bonearr,7,Py_BuildValue("i",bone->childIndex));

			PyList_SetItem(bonearr,8,Py_BuildValue("i",bone->nextSiblingIndex));

			PyList_SetItem(bonearr,9,Py_BuildValue("i",bone->previousSiblingIndex));
					
			PyList_SetItem(self->boneList,i,bonearr);
		}

		//Py_IncRef(self->boneList);
		//printf("[C] Loaded bones! Count: %i\n",self->boneCount);

		//setup the material stuff here wooooo and textures! and mtdparams!!!
		self->materialList = PyList_New(self->asset->header.materialCount);

		for(int i = 0; i < self->asset->header.materialCount; i++)
		{
			PyObject* mat = PyList_New(6);

			PyObject* name = PyUnicode_FromWideChar(self->asset->materials[i]->name,self->asset->materials[i]->nameLength);
			PyList_SetItem(mat,0,name);

			PyObject* mtd = PyUnicode_FromWideChar(self->asset->materials[i]->mtdName,self->asset->materials[i]->mtdNameLength);
			PyList_SetItem(mat,1,mtd);

			PyObject* flags = Py_BuildValue("i",self->asset->materials[i]->header.flags);
			PyList_SetItem(mat,2,flags);

			PyObject* gxIndex = Py_BuildValue("i",self->asset->materials[i]->gxIndex);
			PyList_SetItem(mat,3,gxIndex);
			
			PyObject* texList = PyList_New(self->asset->materials[i]->header.textureCount);
			for(int t = 0; t < self->asset->materials[i]->header.textureCount; t++)
			{
				int texindex = self->asset->materials[i]->header.textureIndex + t;

				PyObject* tex = PyList_New(4);

				PyObject* path = PyUnicode_FromWideChar(self->asset->textures[texindex]->path,self->asset->textures[texindex]->pathLength);
				PyList_SetItem(tex,0,path);

				PyObject* x = Py_BuildValue("f",self->asset->textures[texindex]->scale.x);
				PyList_SetItem(tex,1,x);

				PyObject* y = Py_BuildValue("f",self->asset->textures[texindex]->scale.y);
				PyList_SetItem(tex,2,y);

				PyObject* textype = PyUnicode_FromWideChar(self->asset->textures[texindex]->type,self->asset->textures[texindex]->typeLength);
				PyList_SetItem(tex,3,textype);

				PyList_SetItem(texList,t,tex);
			}
			PyList_SetItem(mat,4,texList);

			PyList_SetItem(self->materialList,i,mat);
		}

		//Py_IncRef(self->materialList);
		//printf("[C] Got %i materials\n",self->materialCount);

		//MESHES
		self->meshList = PyList_New(self->asset->header.meshCount);

		for(int i = 0; i < self->asset->header.meshCount; i++)
		{
			PyObject* mesh = PyList_New(4);

			PyObject* boneIndices = PyList_New(self->asset->meshes[i]->header.boneCount);
			for(int b = 0; b < self->asset->meshes[i]->header.boneCount; b++)
				PyList_SetItem(boneIndices,b,Py_BuildValue("i",self->asset->meshes[i]->boneIndices[b]));
			PyList_SetItem(mesh,0,boneIndices);
			
			PyList_SetItem(mesh,1,Py_BuildValue("i",self->asset->meshes[i]->header.defaultBoneIndex));

			PyList_SetItem(mesh,2,Py_BuildValue("i",self->asset->meshes[i]->header.materialIndex));

			//need this block of code to get the correct faceset to get its information
			cfr::FLVER2::Faceset* facesetp = nullptr;
			cfr::FLVER2::Mesh* meshp = self->asset->meshes[i];
			uint64_t lowest_flags = LLONG_MAX;
			for(int mfsi = 0; mfsi < meshp->header.facesetCount; mfsi++)
			{
				int fsindex = meshp->facesetIndices[mfsi];
				if(self->asset->facesets[fsindex]->header.flags < lowest_flags)
				{
					facesetp = self->asset->facesets[fsindex];
					lowest_flags = facesetp->header.flags;
				}
			}

			PyObject* facesetInfo = PyList_New(2);
			PyList_SetItem(facesetInfo,0,Py_BuildValue("i",facesetp->header.flags));
			PyList_SetItem(facesetInfo,1,Py_BuildValue("b",facesetp->header.cullBackFaces));
			PyList_SetItem(mesh,3,facesetInfo);
			//python stuff expects a list here, so i just did it that way to reduce work

			PyList_SetItem(self->meshList,i,mesh);
		}

		//Py_IncRef(self->meshList);
		//printf("[C] Got %i meshes\n",self->meshCount);

		//FACESETS
		self->facesetList = PyList_New(self->meshCount);

		for(int i = 0; i < self->meshCount; i++)
		{
			cfr::FLVER2::Faceset* facesetp = nullptr;
			cfr::FLVER2::Mesh* meshp = self->asset->meshes[i];
			
			uint64_t lowest_flags = LLONG_MAX;
			
			for(int mfsi = 0; mfsi < meshp->header.facesetCount; mfsi++)
			{
				int fsindex = meshp->facesetIndices[mfsi];
				if(self->asset->facesets[fsindex]->header.flags < lowest_flags)
				{
					facesetp = self->asset->facesets[fsindex];
					lowest_flags = facesetp->header.flags;
				}
			}

			facesetp->triangulate();
			//printf("triangulated %i tris\n",faceset->triCount);

			npy_intp dim[2];
			dim[0] = facesetp->triCount / 3;
			dim[1] = 3;

			PyObject* nparr = PyArray_SimpleNewFromData(2,dim,NPY_INT32,&facesetp->triList[0]);

			//Py_IncRef(nparr);
			PyList_SetItem(self->facesetList,i,nparr);
			
			//free(facesetp->triList);
			//facesetp->triList = NULL;
		}

		//VERTEX BUFFERS
		self->vertexBufferList = PyList_New(self->meshCount);
		self->normalFacesList = PyList_New(self->meshCount); //accelerates settings norms
		
		for(int i = 0; i < self->meshCount; i++)
		{
			cfr::FLVER2::Mesh* mesh = self->asset->meshes[i];

			int vertCount = 0;

			for(int vbi = 0; vbi < self->asset->meshes[i]->header.vertexBufferCount; vbi++) //vertex buffer index
			{
				int vb_index = self->asset->meshes[i]->vertexBufferIndices[vbi];
				vertCount += self->asset->vertexBuffers[vb_index]->header.vertexCount;
			}

			int uvCount = 0;
			int colorCount = 0;
			int tanCount = 0;

			int uvFactor = 1024;
			if(self->asset->header.version >= 0x2000F)
				uvFactor = 2048;

			self->asset->getVertexData(i,&uvCount,&colorCount,&tanCount);

			npy_intp dim1[1] = {vertCount};
			npy_intp dim3[2] = {vertCount,3};
			npy_intp dim4[2] = {vertCount,4};
			npy_intp tanDim[3] = {vertCount,tanCount,3};
			npy_intp uvDim[3] = {vertCount,uvCount,2};
			npy_intp colorDim[3] = {vertCount,colorCount,4};

			PyObject* npPositions = PyArray_SimpleNewFromData(2,dim3,NPY_FLOAT32,mesh->vertexData->positions);
			PyObject* npNoneWeights = PyArray_SimpleNewFromData(2,dim4,NPY_FLOAT32,&mesh->vertexData->bone_weights[0]);
			PyObject* npBoneIndices = PyArray_SimpleNewFromData(2,dim4,NPY_INT16,&mesh->vertexData->bone_indices[0]);
			PyObject* npNormals = PyArray_SimpleNewFromData(2,dim3,NPY_FLOAT32,&mesh->vertexData->normals[0]);
			PyObject* npNormalws = PyArray_SimpleNewFromData(1,dim1,NPY_INT32,&mesh->vertexData->normalws[0]);
			PyObject* npTangents = PyArray_SimpleNewFromData(3,tanDim,NPY_FLOAT32,&mesh->vertexData->tangents[0]);
			PyObject* npBitangets = PyArray_SimpleNewFromData(2,dim4,NPY_FLOAT32,&mesh->vertexData->bitangents[0]);
			PyObject* npUVs = PyArray_SimpleNewFromData(3,uvDim,NPY_FLOAT32,&mesh->vertexData->uvs[0]);
			PyObject* npColors = PyArray_SimpleNewFromData(3,colorDim,NPY_FLOAT32,&mesh->vertexData->colors[0]);

			PyObject* list = PyList_New(9);
			//Py_IncRef(list);
			PyList_SetItem(list,0,npPositions);
			PyList_SetItem(list,1,npNoneWeights);
			PyList_SetItem(list,2,npBoneIndices);
			PyList_SetItem(list,3,npNormals);
			PyList_SetItem(list,4,npNormalws);
			PyList_SetItem(list,5,npTangents);
			PyList_SetItem(list,6,npBitangets);
			PyList_SetItem(list,7,npUVs);
			PyList_SetItem(list,8,npColors);

			PyList_SetItem(self->vertexBufferList,i,list);

			//really quick and dirty way to make faceset indexed normal array
			cfr::FLVER2::Faceset* facesetp = nullptr;
			
			uint64_t lowest_flags = LLONG_MAX;
			
			for(int mfsi = 0; mfsi < mesh->header.facesetCount; mfsi++)
			{
				int fsindex = mesh->facesetIndices[mfsi];
				if(self->asset->facesets[fsindex]->header.flags < lowest_flags)
				{
					facesetp = self->asset->facesets[fsindex];
					lowest_flags = facesetp->header.flags;
				}
			}
			
			//printf("tri count: %i\n",facesetp->triCount);

			facesetp->facesetNorms = (float*)malloc(facesetp->triCount * 3 * 4);
			for(int x = 0; x < facesetp->triCount; x++)
			{
				facesetp->facesetNorms[x] = mesh->vertexData->normals[facesetp->triList[x]];
			}

			npy_intp faceNormDims[2] = {facesetp->triCount,3};
			PyObject* npFaceNorms = PyArray_SimpleNewFromData(2,faceNormDims,NPY_FLOAT32,&facesetp->facesetNorms[0]);
			PyList_SetItem(self->normalFacesList,i,npFaceNorms);
		}

		//printf("[C] Successfully read FLVER2! 0x%x\n",self->asset);
	};

	return 0;
};

PyObject* flverGenerateArmature(flverObject* self, PyObject * args)
{
	return nullptr;
};

PyObject* flverGenerateMesh(flverObject* self, PyObject * args)
{
	static char* kwlist[] = {"mesh","faceset",NULL};
	
	int mesh_i;
	PyObject* bmverts;
	PyObject* bmfaces;
	PyObject* bmloops;
	PyObject* weightLayer;
	PyObject* colorLayers;
	PyObject* uvLayers;
	
	if(!PyArg_ParseTuple(args,"iOOOOOO",&mesh_i,&bmverts,&bmfaces,&bmloops,&weightLayer,&colorLayers,&uvLayers))
	{
		return NULL;
	}

	cfr::FLVER2::Mesh* mesh = self->asset->meshes[mesh_i];
	cfr::FLVER2::Faceset* facesetp = NULL;
	
	uint64_t lowest_flags = LLONG_MAX;
	for(int mfsi = 0; mfsi < mesh->header.facesetCount; mfsi++)
	{
		int fsindex = mesh->facesetIndices[mfsi];
		if(self->asset->facesets[fsindex]->header.flags < lowest_flags)
		{
			facesetp = self->asset->facesets[fsindex];
			lowest_flags = facesetp->header.flags;
		}
	}
	
	//printf("Got best faceset... ");

	int uvCount = PyList_Size(uvLayers);
	int colorCount = PyList_Size(colorLayers);

	bool useMeshBones = false;
	if(mesh->header.boneCount > 0)
		useMeshBones = true;

	int vertCount = 0;
	PyObject* vertArr[3];
	std::map<int,PyObject*> bmVertMap;

	//printf("Beginning faceset loop!\n");

	for(int i = 0; i < facesetp->triCount; i++)
	{
		int vertexIndex = facesetp->triList[i];

		PyObject* vert;

		//if vert not already created and in map, create new and setup
		if(!bmVertMap.contains(vertexIndex))
		{
			float x = mesh->vertexData->positions[(vertexIndex*3)+0];
			float y = mesh->vertexData->positions[(vertexIndex*3)+1];
			float z = mesh->vertexData->positions[(vertexIndex*3)+2];
			//create new bmvert object
			//PyObject* newVertMeth = PyObject_GetAttrString(bmverts, "new");
			//PyObject *args = Py_BuildValue("(fff)",x,y,z);    

			vert = PyObject_CallMethod(bmverts,"new","[fff]",x,y,z);
			//apply bone weights and indices
			for(int b = 0; b < 4; b++)
			{

				int bone_index = mesh->vertexData->bone_indices[(vertexIndex*4)+b];
				float bone_weight= mesh->vertexData->bone_weights[(vertexIndex*4)+b];

				if((b == 0) && (bone_weight <= 0.001f))
				{
					bone_weight = 1.0f;
				}

				if(bone_weight > 0.01f) //culling "unused" bones
				{
					PyObject* py_bone_weight = Py_BuildValue("f",bone_weight);
					PyObject* vert_weight = PyObject_GetItem(vert,weightLayer);

					if(useMeshBones) //ptde always indexes bones based on the mesh bone indices
					{
						PyObject* py_bone_index = Py_BuildValue("i",mesh->boneIndices[bone_index]);
						PyObject_SetItem(vert_weight,py_bone_index,py_bone_weight);
					}
					else if(bone_index >= 0) //ds3 tends to directly index the flver bones
					{
						PyObject* py_bone_index = Py_BuildValue("i",bone_index);
						PyObject_SetItem(vert_weight,py_bone_index,py_bone_weight);
					}
					//PyErr_Restore(NULL,NULL,NULL);
				}
			}

			//TODO: add in colors

			//add to map
			bmVertMap.insert(std::pair<int,PyObject*>(vertexIndex,vert));
		}
		else
		{
			vert = bmVertMap.at(vertexIndex);
		}

		vertArr[vertCount] = vert;
		vertCount++;

		//make face from every three verts
		if(vertCount == 3)
		{
			vertCount = 0; //reset counter

			//create face object from last three vert pyobjects
			PyObject* face = PyObject_CallMethod(bmfaces,"new","[OOO]",vertArr[0],vertArr[1],vertArr[2]);

			//TODO: check for error, if no error, do uv stuff
			if(PyErr_Occurred() == NULL)
			{
				PyObject* loop_attr = PyObject_GetAttrString(face,"loops");
				PyObject* face_loops = PyObject_GetAttr(face,loop_attr);
				printf("got face loop object\n");
				for(int uli = 0; uli < uvCount; uli++)
				{
					for(int l = 0; l < 3; l++)
					{
						printf("uli: %i\n",uli);
						printf("l: %i\n",l);
						PyObject* loop = PyObject_GetItem(face_loops,Py_BuildValue("i",l));
						//PyObject* loop = PyList_GET_ITEM(face_loops,l);
						PyObject* uv_layer = PyList_GetItem(uvLayers,uli);
						if(uv_layer == NULL)
						{
							printf("AUGUGUGUGGUGGG\n");
						}
						printf("hw\n");
						//PyObject* loopUV = PyMapping_HasKey(loop,uv_layer);
						int loopUV = PyMapping_HasKey(loop,uv_layer);
						if(loopUV == 0)
						{
							printf("AAAAAAAAAA\n");
						}
						printf("hw\n");

						//PyObject* face_loop_uv = PyObject_GetAttrString(loopUV,"uv");
						printf("got all uv type things, time to set attr");
						/*int uv_offset = ((vertexIndex-3+l)*(2*uvCount))+uli;
						float uv_x = mesh->vertexData->uvs[uv_offset+0];
						float uv_y = mesh->vertexData->uvs[uv_offset+1];*/
						PyObject* lazyUvs = Py_BuildValue("ff",0,0);
						PyObject_SetAttrString(face_loops,"uv",lazyUvs);
					}
				}
			}
		}
	}

	PyObject* nothing = Py_BuildValue("i",0);
	return nothing;
};

PyObject* flverGetFaceset(flverObject* self, PyObject *args)
{
	/*static char* kwlist[] = {"mesh","faceset",NULL};
	
	int mesh_i;

	if(!PyArg_ParseTuple(args,"i",&mesh_i))
	{
		return NULL;
	}

	cfr::FLVER2::Faceset* facesetp = nullptr;
	cfr::FLVER2::Mesh* meshp = self->asset->meshes[mesh_i];

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

	return nparr;*/
	return NULL;
};

//returns list of nparrys
PyObject* flverGetVertData(flverObject* self, PyObject *args)
{
	return NULL;
};

//more data, but might be faster since less jumping around?
PyObject* flverGetVertDataOrdered(flverObject* self, PyObject *args)
{
	return NULL;
};

int flverClose(flverObject* self, PyObject *args)
{
	/*int i = 100;
	while(i > 0)
	{
		Py_XDECREF(self->boneList);
		Py_XDECREF(self->meshList);
		Py_XDECREF(self->materialList);
		Py_XDECREF(self->dummyList);
		Py_XDECREF(self->filePath);
		i--;
	}

	self->asset->forceClose();
	delete self->asset;*/

	return 0;
};

/*static PyObject* flverExportGLTF(flverObject* self, PyObject *args)
{
	cfr::exportGLTF(self->asset);

	return Py_BuildValue("i",0);
};*/
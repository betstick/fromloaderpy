
import fromloader
import shutil
import sys
import os
import faulthandler
from os.path import exists
from typing import Any, cast

def get_asset_paths(id):
	map_path = ds3_path + "/map/" + id + "/"
	map_files = os.listdir(map_path)
	flver_paths = []
		
	for file in map_files:
		if  "-mapbnd-dcx" in file:
			fname = file.replace("-mapbnd-dcx","")
			flver_paths.append(map_path+file+"/map/"+id+"/"+fname+"/Model/"+fname+".flver")
		
	return flver_paths

ptde_path = 'C:\\Program Files (x86)\\Steam\\steamapps\\common\\Dark Souls Prepare to Die Edition\\DATA\\'

#fromloader.mtd("A05_1Wood[DSB].mtd")
gwyn = ptde_path + 'chr\\c5370-chrbnd\\chr\\c5370\\c5370.flver'
#flver = fromloader.flver2(gwyn)
#print(flver.material_count)
#print(flver.bones[0])
#print(flver.meshes[0])
#print(flver.materials[0][1].split("\\")[-1])

ds3_path = 'D:/SteamLibrary/steamapps/common/DARK SOULS III/Game'
map_path = ''


#mtdPath = ptde_path + 'mtd\\Mtd-mtdbnd\\A10_Stone[DSB].mtd'
badmtd = 'D:/SteamLibrary/steamapps/common/DARK SOULS III/Game/mtd/allmaterialbnd-mtdbnd-dcx/M[ARSN]_2m_l.mtd'
badflver = 'D:/SteamLibrary/steamapps/common/DARK SOULS III/Game/map/m40_00_00_00/m40_00_00_00_004310-mapbnd-dcx/map/m40_00_00_00/m40_00_00_00_004310/Model/m40_00_00_00_004310.flver'
ptde_mtd_path = '/mtd/Mtd-mtdbnd/'
ds3_mtd_path = '/mtd/allmaterialbnd-mtdbnd-dcx/'

#flver_paths = get_asset_paths("m40_00_00_00")

#for i in range(1024):
#	print(i)

faulthandler.enable(all_threads=True)
f = fromloader.flver2(gwyn)

u = fromloader.util()
u.print_custom_data_layer(f)

print(f.bone_count)
exit()
fi = 0
#for path in flver_paths:
for i in range(1):
	#print(str(fi) + " " + path.split("/")[-1])
	#f = fromloader.flver2(path)
	f = fromloader.flver2(gwyn)

	print(len(f.face_normals))
	#print(len(f.materials))
	"""for mat in f.materials:
		#print(mat)
		mtd_name = mat[1].split("\\")[-1].replace("\0","")
		#print("\t" + mtd_name)
		mtd_path = (ptde_path + ptde_mtd_path + mtd_name).replace("\\","/")
		#print(mtd_path)
		if exists(mtd_path) and (os.path.isfile(mtd_path)):
			mtd = fromloader.mtd(mtd_path)
			for param in mtd.params:
				print("Param: " + str(param[0]))"""
	print(f.face_normals[0])
	fi += 1

	#imp.reload(fromloader)
	#mtd = fromloader.mtd(badmtd)
	#g = fromloader.flver2(gwyn)
	#f = fromloader.flver2(badflver)
	#del mtd

print("AAAA")
from skbuild import setup
import shutil, os

setup(
	name='fromloader-cpp',
	packages=['fromloader'],
	version="1.2.3",
	#package_dir={'': 'src'},
)

#need to copy these files cause windows 10 blender/python interaction is bad
#shutil.copy(,)

#python file handling is absolute pain
try:
	files = os.listdir("..\\fromloader")
	for file in files:
		try:
			os.remove("..\\fromloader\\" + file)
		except:
			None
	try:
		cache = os.listdir("..\\fromloader\\__pycache__")
		for file in cache:
			try:
				os.remove("..\\fromloader\\__pycache__\\" + file)
			except:
				None
		try:
			os.removedirs("..\\fromloader\\__pycache__")
		except:
			None
	except:
		None
	try:
		os.removedirs("..\\fromloader")
	except:
		None
except:
	None

#copy the addon to the correct location
try:
	shutil.copytree(
		os.path.join("_skbuild\\win-amd64-3.10\\setuptools\\lib\\","fromloader"),
		os.path.join("..\\fromloader")
	)
except:
	print("Failed to copy addon files to upper folder!")
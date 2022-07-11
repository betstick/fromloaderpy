#apparently this is depricated? TODO fix later
from distutils.file_util import copy_file

import os
import platform
import shutil
import subprocess
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

c_module_name = 'fromloader'

cmake_cmd_args = []

for f in sys.argv:
	if f.startswith('-D'):
		cmake_cmd_args.append(f)

for f in cmake_cmd_args:
	sys.argv.remove(f)

def _get_env_variable(name, default='OFF'):
	if name not in os.environ.keys():
		return default
	return os.environ[name]

class CMakeExtension(Extension):
	def __init__(self,name,cmake_lists_dir='.',**kwa):
		Extension.__init__(self,name,sources=[],**kwa)
		self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)

class CMakeBuild(build_ext):
	def build_extensions(self):
		try:
			out = subprocess.check_output(['cmake', '--version'])
		except OSError:
			raise RuntimeError('Cannot find CMake executable')

		for ext in self.extensions:

			extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
			cfg = 'Release' #if _get_env_variable('DISPTOOLS_DEBUG') == 'ON' else 'Release'

			cmake_args = []

			if platform.system() == 'Windows':
				plat = ('x64' if platform.architecture()[0] == '64bit' else 'Win32')
				cmake_args += [
					#'/std=c++23',
					'-DVCPKG_BUILD_TYPE=release',
					'-DCMAKE_BUILD_TYPE=Release',
					'-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE',
					'-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir),
				]
				if self.compiler.compiler_type == 'msvc':
					cmake_args += [
						'-DCMAKE_GENERATOR_PLATFORM=%s' % plat,
					]
				else:
					cmake_args += [
						'-G', 'MinGW Makefiles',
					]

			cmake_args += cmake_cmd_args

			#pprint(cmake_args)

			if not os.path.exists(self.build_temp):
				os.makedirs(self.build_temp)

			# Config and build the extension
			subprocess.check_call(['cmake', ext.cmake_lists_dir] + cmake_args,
								cwd=self.build_temp)
			subprocess.check_call(['cmake', '--build', '.', '--config', cfg],
								cwd=self.build_temp)

setup(
	name='fromloader',
	#packages=['fromloader'],
	version='1.0.0',
	ext_modules=[CMakeExtension('fromloader')],
	cmdclass={'build_ext': CMakeBuild},
	zip_safe=False,
)
try:
	copy_file('build/lib.win-amd64-3.10/fromloader.pyd','..\\fromloader_d\\')
	copy_file("build/lib.win-amd64-3.10/python310.dll","..\\fromloader_d\\")
	copy_file("build/lib.win-amd64-3.10/zlib1.dll","..\\fromloader_d\\")
except:
	print("failed to copy the files. close blender")
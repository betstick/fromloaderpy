cmake_minimum_required(VERSION 3.18)
project(fromloader)

include(GNUInstallDirs)

find_package(ZLIB REQUIRED)
include_directories(${ZLIB_INCLUDE_DIR})
include_directories(${ZLIB_INCLUDE_DIRS})

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)

find_package(PythonExtensions REQUIRED)

execute_process(
  COMMAND "${PYTHON_EXECUTABLE}"
  -c "import numpy; print(numpy.get_include())"
  OUTPUT_VARIABLE NumPy_INCLUDE_DIRS
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
# F2PY headers
execute_process(
  COMMAND "${PYTHON_EXECUTABLE}"
  -c "import numpy.f2py; print(numpy.f2py.get_include())"
  OUTPUT_VARIABLE F2PY_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)


if(WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++20 /Zc:strictStrings- ")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /std:c++20 /Zc:strictStrings- ")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wl,--no-as-needed")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -Wl,--no-as-needed")
endif()

if(MSVC)
	if(BUILD STREQUAL "release")
		set(VCPKG_BUILD_TYPE release)
	else()
		set(VCPKG_BUILD_TYPE debug)
	endif()
endif()

if(NOT BUILD)
	set(BUILD "release")
endif()

if(BUILD STREQUAL "release")
	set(BUILD "Release")
	#optimize aggressively
	if(MSVC)
		set(VCPKG_BUILD_TYPE release)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}/O2")
	else()
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
	endif()
	add_compile_definitions(RELEASE)
	message(STATUS "Using release mode...")
else()
	set(BUILD "Debug")
	
	#show all types of errors, but limit to 8 at a time
	if(MSVC)
		set(VCPKG_BUILD_TYPE debug)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++20 /Zc:strictStrings- /O0")
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /std:c++20 /Zc:strictStrings- /O0")
	else()
		#enable GDB and disable all optimizations
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -O0 -g")
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ggdb -O0 -g")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -fmax-errors=8")
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Werror -fmax-errors=8")
	endif()
	#enable this for my debug code preprocessor stuff
	
	#needed to prevent VMA werror stuff from stopping compile.
	add_compile_definitions(VMA_DEBUG_DETECT_CORRUPTION)
	add_compile_definitions(VMA_DEBUG_MARGIN=1)
	message(STATUS "Using debug mode...")
endif()

add_subdirectory(lib/cmem/)
add_subdirectory(lib/umem/)
add_subdirectory(lib/fromloader/)

set(SOURCES
	#module.cpp
	fromloader/_fromloader.cpp
	fromloader/flver2/flver2.cpp
	#flver2/bone/bone.cpp
	fromloader/mtd/mtd.cpp
)

add_library(_fromloader MODULE ${SOURCES})

target_link_libraries(_fromloader frld umem cmem ZLIB::ZLIB)

target_include_directories(_fromloader PUBLIC
	${NumPy_INCLUDE_DIRS}
	${PYTHON_INCLUDE_DIRS}
	cmem
	umem
	frld
)

python_extension_module(_fromloader)

install(TARGETS _fromloader DESTINATION fromloader)

unset(CMAKE_BUILD_TYPE CACHE)
message(STATUS "Given CMEM_LOCATION: " ${CMEM_DIR})

project(frompy-proj)

if(MSVC)
	if(BUILD STREQUAL "release")
		set(VCPKG_BUILD_TYPE release)
	else()
		set(VCPKG_BUILD_TYPE debug)
	endif()
endif()

find_package(ZLIB REQUIRED)
include_directories(${ZLIB_INCLUDE_DIR})
include_directories(${ZLIB_INCLUDE_DIRS})

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(Python COMPONENTS Interpreter Development.Module NumPy)
find_package(Python3 COMPONENTS Interpreter NumPy REQUIRED)

# this apparently adds the numpy stuff
exec_program(${PYTHON_EXECUTABLE}
             ARGS "-c \"import numpy; print(numpy.get_include())\""
             OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
             RETURN_VALUE NUMPY_NOT_FOUND
            )
if(NUMPY_NOT_FOUND)
    message(FATAL_ERROR "NumPy headers not found")
endif()

include_directories(cmem)
include_directories(umem)
include_directories(frld)
include_directories(${PYTHON_INCLUDE_DIRS})

set(SOURCES
	module.cpp
	flver2/flver2.cpp
	#flver2/bone/bone.cpp
	mtd/mtd.cpp
)


add_library(frompy-proj SHARED ${SOURCES})

target_include_directories(frompy-proj PUBLIC
	${PYTHON_INCLUDE_DIRS}
	cmem
	umem
	frld
)

set_target_properties ( frompy-proj PROPERTIES
	OUTPUT_NAME "fromloader"
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" 
	SUFFIX ".pyd"
)

if(MSVC)
	if(BUILD STREQUAL "release")

	else()
		target_link_options(frompy-proj PUBLIC /DEBUG)
	endif()
endif()

target_link_libraries(frompy-proj frld umem cmem ZLIB::ZLIB ${PYTHON_LIBRARIES} Python3::NumPy)

#set_target_properties ( cfr-test PROPERTIES
#	OUTPUT_NAME "flver"
#	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" 
#)

#target_link_libraries(cfr-test frld umem cmem ZLIB::ZLIB PythonInterp PythonLibs)
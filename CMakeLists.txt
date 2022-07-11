cmake_minimum_required(VERSION 3.10)

project(outer-proj)

include(GNUInstallDirs)

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)

if(WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++20 /Zc:strictStrings- ")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /std:c++20 /Zc:strictStrings- ")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
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

set(CMAKE_BINARY_DIR bin/)

add_subdirectory(lib/cmem/)
add_subdirectory(lib/umem/)
add_subdirectory(lib/fromloader/)

add_subdirectory(src/)

unset(CMAKE_BUILD_TYPE CACHE)
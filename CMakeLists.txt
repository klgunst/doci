cmake_minimum_required(VERSION 3.7.2)
project(doci)
set(doci_MMP_VERSION "0.1")
set(doci_VERSION "${doci_MMP_VERSION} (2017-05-28)")
set(doci_AUTHORS "Klaas Gunst")
set(doci_DESCRIPTION "An implementation of DOCI (doubly occupied configuration interaction)")
set(doci_EMAIL "klaas.gunst@ugent.be")

#Bring the headers into the project
include_directories(include)

#Add the sources
file(GLOB SOURCES "src/*.c")

option(BUILD_DOXYGEN	"Use Doxygen to create a HTML/PDF manual" ON)

# <<<  Find LAPACK and BLAS  >>>
find_package (LAPACK REQUIRED)

# OpenMP
find_package (OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif (OPENMP_FOUND)

# debug symbols
# set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")

# highly optimized
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")


if (BUILD_DOXYGEN)
    find_package (Doxygen)
    if (NOT DOXYGEN_FOUND)
	message (FATAL_ERROR "Doxygen is needed to build the documentation. Please install it correctly.")
    endif(NOT DOXYGEN_FOUND)
    configure_file (${doci_SOURCE_DIR}/src/Doxyfile.in ${doci_BINARY_DIR}/Doxyfile @ONLY IMMEDIATE)
    add_custom_target (doc COMMAND ${DOXYGEN_EXECUTABLE} ${doci_BINARY_DIR}/Doxyfile SOURCES ${doci_BINARY_DIR}/Doxyfile)
endif(BUILD_DOXYGEN)

add_executable(doci ${SOURCES})
target_link_libraries(doci ${LAPACK_LIBRARIES})

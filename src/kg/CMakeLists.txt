
add_library(kg INTERFACE)
target_compile_features(kg INTERFACE cxx_std_11)
target_include_directories(kg INTERFACE include)
# FIXME, hack for mrc_common.h
target_include_directories(kg INTERFACE ${CMAKE_SOURCE_DIR}/src/libmrc/include)
# FIXME, hack for vec3.h; PscConfig.h
target_include_directories(kg INTERFACE ${CMAKE_SOURCE_DIR}/src/include)
target_link_libraries(kg INTERFACE MPI::MPI_CXX)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(kg INTERFACE OpenMP::OpenMP_CXX)
endif()
if (PSC_HAVE_ADIOS2)
  target_link_libraries(kg INTERFACE adios2::adios2)
endif()

if (BUILD_TESTING)
  add_subdirectory(testing)
endif()

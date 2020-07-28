
find_library(RMM_LIBRARIES NAMES rmm)

find_path(RMM_INCLUDE_DIRS NAMES rmm/thrust_rmm_allocator.h)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(RMM
  REQUIRED_VARS RMM_LIBRARIES RMM_INCLUDE_DIRS
)

if (RMM_FOUND)
    if(NOT TARGET RMM::RMM)
      add_library(RMM::RMM INTERFACE IMPORTED)
      set_property(TARGET RMM::RMM PROPERTY
        INTERFACE_INCLUDE_DIRECTORIES "${RMM_INCLUDE_DIRS}")
      set_property(TARGET RMM::RMM PROPERTY
        INTERFACE_LINK_LIBRARIES "${RMM_LIBRARIES}")
    endif()
endif()


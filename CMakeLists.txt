cmake_minimum_required(VERSION 3.15)
project(nominate_cmodule)

# Forzar modo Release por defecto
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Buscar dependencias
find_package(OpenMP REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(NLOPT REQUIRED nlopt)
find_package(BLAS)

# Incluir directorios
include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/Eigen
    ${NLOPT_INCLUDE_DIRS}
)

# Fuentes compartidas
set(CORE_SOURCES
    src/dwnominate.cpp
    src/csv_loader.cpp
    src/likelihood.cpp
    src/normal_cdf.cpp
    src/cutting_plane.cpp
    src/cutting_point.cpp
    src/grid_optimizer.cpp
    src/rollcall_optimizer.cpp
    src/rollcall_derivatives.cpp
    src/optimize_legislators.cpp
    src/legislator_derivatives.cpp
    src/simple_ols.cpp
)

# EJECUTABLE PRINCIPAL: dwnominate
add_executable(dwnominate src/main_cli.cpp ${CORE_SOURCES})

target_link_libraries(dwnominate 
    ${NLOPT_LIBRARIES}
    OpenMP::OpenMP_CXX
)
target_link_directories(dwnominate PRIVATE ${NLOPT_LIBRARY_DIRS})
target_compile_definitions(dwnominate PRIVATE USE_OPENMP)
target_compile_options(dwnominate PRIVATE -O3 -march=native -DNDEBUG)

if(BLAS_FOUND)
    target_link_libraries(dwnominate ${BLAS_LIBRARIES})
    target_compile_definitions(dwnominate PRIVATE EIGEN_USE_BLAS)
endif()

# VALIDADOR CSV: test_validate_csv
add_executable(test_validate_csv tests/test_validate_csv.cpp)
target_compile_options(test_validate_csv PRIVATE -O3 -DNDEBUG)

# Mensajes de estado
message(STATUS "NLopt: ${NLOPT_LIBRARIES}")
message(STATUS "OpenMP: ${OpenMP_CXX_VERSION}")
if(BLAS_FOUND)
    message(STATUS "BLAS: ${BLAS_LIBRARIES}")
endif()


cmake_minimum_required(VERSION 3.15)
project(nominate_cmodule)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Mostrar información del compilador
message(STATUS "C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C Compiler Version: ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "C++ Compiler Version: ${CMAKE_CXX_COMPILER_VERSION}")

# Buscar OpenMP
find_package(OpenMP REQUIRED)

# Buscar NLopt (desde MSYS2)
find_package(PkgConfig REQUIRED)
pkg_check_modules(NLOPT REQUIRED nlopt)

# Incluir directorios
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)  # nlohmann/json.hpp está aquí
include_directories(${CMAKE_SOURCE_DIR}/Eigen)    # Eigen headers
include_directories(${NLOPT_INCLUDE_DIRS})

# Verificar que json.hpp existe
if(EXISTS "${CMAKE_SOURCE_DIR}/include/json.hpp")
    message(STATUS "nlohmann/json encontrado en include/json.hpp")
else()
    message(WARNING "nlohmann/json NO encontrado. Descarga json.hpp a include/")
endif()

# ==============================================================
# EJECUTABLE PRINCIPAL: nominate_cmodule (programa DW-NOMINATE)
# ==============================================================
add_executable(nominate_cmodule 
    src/main.cpp
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

# Enlazar librerías con NLopt de MSYS2
target_link_libraries(nominate_cmodule ${NLOPT_LIBRARIES})
target_link_directories(nominate_cmodule PRIVATE ${NLOPT_LIBRARY_DIRS})

add_executable(test_json tests/test_json.cpp)
# test_json no necesita NLopt ni OpenMP, solo los headers

add_executable(test_normal_cdf 
    tests/test_normal_cdf.cpp
    src/normal_cdf.cpp
)
target_compile_features(test_normal_cdf PRIVATE cxx_std_17)

add_executable(test_likelihood
    tests/test_likelihood.cpp
    src/likelihood.cpp
    src/normal_cdf.cpp
)
target_compile_features(test_likelihood PRIVATE cxx_std_17)

add_executable(test_sort_utils
    tests/test_sort_utils.cpp
)
target_compile_features(test_sort_utils PRIVATE cxx_std_17)

add_executable(test_grid_optimizer
    tests/test_grid_optimizer.cpp
    src/grid_optimizer.cpp
    src/likelihood.cpp
    src/normal_cdf.cpp
)
target_compile_features(test_grid_optimizer PRIVATE cxx_std_17)

# Test de comparacion SIGMAS (validacion cruzada contra Fortran)
add_executable(test_sigmas_comparison
    tests/test_sigmas_comparison.cpp
)
target_compile_features(test_sigmas_comparison PRIVATE cxx_std_17)

add_executable(test_wint
    tests/test_wint.cpp
    src/grid_optimizer.cpp
    src/likelihood.cpp
    src/normal_cdf.cpp
)
target_compile_features(test_wint PRIVATE cxx_std_17)


add_executable(test_rollcall_derivatives
    tests/test_rollcall_derivatives.cpp
    src/rollcall_derivatives.cpp
    src/likelihood.cpp
    src/normal_cdf.cpp
)
target_compile_features(test_rollcall_derivatives PRIVATE cxx_std_17)

add_executable(test_rollcall_optimizer
    tests/test_rollcall_optimizer.cpp
    src/rollcall_optimizer.cpp
    src/rollcall_derivatives.cpp
    src/likelihood.cpp
    src/normal_cdf.cpp
)
target_compile_features(test_rollcall_optimizer PRIVATE cxx_std_17)

add_executable(test_rcint2_crossvalidation
    tests/test_rcint2_crossvalidation.cpp
    src/rollcall_optimizer.cpp
    src/rollcall_derivatives.cpp
    src/likelihood.cpp
    src/normal_cdf.cpp
)
target_compile_features(test_rcint2_crossvalidation PRIVATE cxx_std_17)

add_executable(test_legislator_derivatives
    tests/test_legislator_derivatives.cpp
    src/legislator_derivatives.cpp
    src/likelihood.cpp
    src/normal_cdf.cpp
)
target_compile_features(test_legislator_derivatives PRIVATE cxx_std_17)

add_executable(test_prox_crossvalidation
    tests/test_prox_crossvalidation.cpp
    src/legislator_derivatives.cpp
    src/likelihood.cpp
    src/normal_cdf.cpp
)
target_compile_features(test_prox_crossvalidation PRIVATE cxx_std_17)

add_executable(test_rega_crossvalidation
    tests/test_rega_crossvalidation.cpp
    src/simple_ols.cpp
)
target_compile_features(test_rega_crossvalidation PRIVATE cxx_std_17)

add_executable(test_xint_crossvalidation
    tests/test_xint_crossvalidation.cpp
    src/optimize_legislators.cpp
    src/legislator_derivatives.cpp
    src/simple_ols.cpp
    src/likelihood.cpp
    src/normal_cdf.cpp
)
target_compile_features(test_xint_crossvalidation PRIVATE cxx_std_17)

# Test End-to-End Chile (C++ vs R con datos reales)
add_executable(test_chile_endtoend
    tests/test_chile_endtoend.cpp
    src/csv_loader.cpp
    src/dwnominate.cpp
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
target_compile_features(test_chile_endtoend PRIVATE cxx_std_17)

# Configuración OpenMP
if(OpenMP_CXX_FOUND)
    # Enlazar OpenMP a ambos ejecutables
    target_link_libraries(nominate_cmodule OpenMP::OpenMP_CXX)
    target_link_libraries(test_chile_endtoend OpenMP::OpenMP_CXX)
    target_compile_definitions(nominate_cmodule PRIVATE USE_OPENMP)
    target_compile_definitions(test_chile_endtoend PRIVATE USE_OPENMP)
    message(STATUS "OpenMP ${OpenMP_CXX_VERSION} encontrado y configurado")
    message(STATUS "OpenMP flags: ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "OpenMP no encontrado. Verifica instalación MSYS2.")
endif()

target_compile_features(nominate_cmodule PRIVATE cxx_std_14)

# Configuración optimizada para ambos ejecutables
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(nominate_cmodule PRIVATE -O3 -march=native)
    target_compile_options(test_chile_endtoend PRIVATE -O3 -march=native -DNDEBUG)
endif()

# Información de NLopt
message(STATUS "NLopt encontrado:")
message(STATUS "Include dirs: ${NLOPT_INCLUDE_DIRS}")
message(STATUS "Library dirs: ${NLOPT_LIBRARY_DIRS}")
message(STATUS "Libraries: ${NLOPT_LIBRARIES}")

# Añadir soporte OpenBLAS
find_package(BLAS)
if(BLAS_FOUND)
    message(STATUS "BLAS encontrado: ${BLAS_LIBRARIES}")
    target_link_libraries(nominate_cmodule ${BLAS_LIBRARIES})
    target_link_libraries(test_chile_endtoend ${BLAS_LIBRARIES})
    target_compile_definitions(nominate_cmodule PRIVATE EIGEN_USE_BLAS)
    target_compile_definitions(test_chile_endtoend PRIVATE EIGEN_USE_BLAS)
endif()

# Flags de optimización agresiva
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(nominate_cmodule PRIVATE 
        -O3 -march=native -ffast-math -DNDEBUG)
    target_compile_options(test_chile_endtoend PRIVATE 
        -O3 -march=native -ffast-math -DNDEBUG)
endif()

# Test CUTPLANE Debug (sin OpenMP para depuración)
add_executable(test_cutplane_debug
    tests/test_cutplane_debug.cpp
    src/csv_loader.cpp
    src/dwnominate.cpp
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
target_compile_features(test_cutplane_debug PRIVATE cxx_std_17)

# NO añadir OpenMP para este test de debug
if(BLAS_FOUND)
    target_link_libraries(test_cutplane_debug ${BLAS_LIBRARIES})
    target_compile_definitions(test_cutplane_debug PRIVATE EIGEN_USE_BLAS)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(test_cutplane_debug PRIVATE -O2 -DNDEBUG)
endif()
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

# EJECUTABLE CLI: dwnominate (interfaz de línea de comandos)
add_executable(dwnominate 
    src/main_cli.cpp
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
target_compile_features(dwnominate PRIVATE cxx_std_17)

# Enlazar librerías con NLopt de MSYS2

target_link_libraries(dwnominate ${NLOPT_LIBRARIES})
target_link_directories(dwnominate PRIVATE ${NLOPT_LIBRARY_DIRS})

add_executable(test_json tests/test_json.cpp)
# test_json no necesita NLopt ni OpenMP, solo los headers

# Test End-to-End Chile (C++ vs R con datos reales)
add_executable(test_chile_endtoend
    tests/test_chile_endtoend_initwnominate.cpp
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

# Test End-to-End Chile WNOMINATE (iniciando desde coordenadas WNOMINATE)
add_executable(test_chile_endtoend_copy
    tests/test_chile_endtoend_initwnominate_copy.cpp
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
target_compile_features(test_chile_endtoend_copy PRIVATE cxx_std_17)

# VALIDADOR CSV: compara resultados C++ vs R
add_executable(test_validate_csv
    tests/test_validate_csv.cpp
)
target_compile_features(test_validate_csv PRIVATE cxx_std_17)

# Configuración OpenMP
if(OpenMP_CXX_FOUND)
    # Enlazar OpenMP a todos los ejecutables
    target_link_libraries(dwnominate OpenMP::OpenMP_CXX)
    target_link_libraries(test_chile_endtoend OpenMP::OpenMP_CXX)
    target_link_libraries(test_chile_endtoend_copy OpenMP::OpenMP_CXX)
    target_compile_definitions(dwnominate PRIVATE USE_OPENMP)
    target_compile_definitions(test_chile_endtoend PRIVATE USE_OPENMP)
    target_compile_definitions(test_chile_endtoend_copy PRIVATE USE_OPENMP)
    message(STATUS "OpenMP ${OpenMP_CXX_VERSION} encontrado y configurado")
    message(STATUS "OpenMP flags: ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "OpenMP no encontrado. Verifica instalación MSYS2.")
endif()


# Configuración optimizada para ambos ejecutables
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(dwnominate PRIVATE -O3 -march=native)
    target_compile_options(test_chile_endtoend PRIVATE -O3 -march=native -DNDEBUG)
    target_compile_options(test_chile_endtoend_copy PRIVATE -O3 -march=native -DNDEBUG)
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
    target_link_libraries(dwnominate ${BLAS_LIBRARIES})
    target_link_libraries(test_chile_endtoend ${BLAS_LIBRARIES})
    target_link_libraries(test_chile_endtoend_copy ${BLAS_LIBRARIES})
    target_compile_definitions(dwnominate PRIVATE EIGEN_USE_BLAS)
    target_compile_definitions(test_chile_endtoend PRIVATE EIGEN_USE_BLAS)
    target_compile_definitions(test_chile_endtoend_copy PRIVATE EIGEN_USE_BLAS)
endif()

# Flags de optimización agresiva
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(dwnominate PRIVATE 
        -O3 -march=native -ffast-math -DNDEBUG)
    target_compile_options(test_chile_endtoend PRIVATE 
        -O3 -march=native -ffast-math -DNDEBUG)
    target_compile_options(test_chile_endtoend_copy PRIVATE 
        -O3 -march=native -ffast-math -DNDEBUG)
endif()

cmake_minimum_required(VERSION 3.15)
project(nominate_cmodule)

set(CMAKE_CXX_STANDARD 14)
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
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${NLOPT_INCLUDE_DIRS})

# Buscar archivos fuente
file(GLOB CXX_SOURCES "src/*.cpp")

# Crear ejecutable
add_executable(nominate_cmodule ${CXX_SOURCES})

# Enlazar librerías con NLopt de MSYS2
target_link_libraries(nominate_cmodule ${NLOPT_LIBRARIES})
target_link_directories(nominate_cmodule PRIVATE ${NLOPT_LIBRARY_DIRS})

# Configuración OpenMP
if(OpenMP_CXX_FOUND)
    target_link_libraries(nominate_cmodule OpenMP::OpenMP_CXX)
    target_compile_definitions(nominate_cmodule PRIVATE USE_OPENMP)
    message(STATUS "OpenMP ${OpenMP_CXX_VERSION} encontrado y configurado")
    message(STATUS "OpenMP flags: ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "OpenMP no encontrado. Verifica instalación MSYS2.")
endif()

target_compile_features(nominate_cmodule PRIVATE cxx_std_14)

# Configuración optimizada
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(nominate_cmodule PRIVATE -O3 -march=native)
endif()

# Información de NLopt
message(STATUS "NLopt encontrado:")
message(STATUS "Include dirs: ${NLOPT_INCLUDE_DIRS}")
message(STATUS "Library dirs: ${NLOPT_LIBRARY_DIRS}")
message(STATUS "Libraries: ${NLOPT_LIBRARIES}")
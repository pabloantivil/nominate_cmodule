cmake_minimum_required(VERSION 3.15)

# Nombre del proyecto
project(nominate_cmodule)

# CAMBIAR A C++14 para compatibilidad
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Añadir flags específicos para MinGW
if(MINGW)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
endif()

# Incluir Eigen
include_directories(${CMAKE_SOURCE_DIR})

# Incluir NLopt
include_directories(${CMAKE_SOURCE_DIR}/nlopt/include)
link_directories(${CMAKE_SOURCE_DIR}/nlopt/lib)

# Buscar todos los .cpp en src/
file(GLOB CXX_SOURCES "src/*.cpp")

# Crear ejecutable
add_executable(nominate_cmodule ${CXX_SOURCES})

# Enlazar con NLopt
target_link_libraries(nominate_cmodule nlopt)

# Asegurar que el target use C++14
target_compile_features(nominate_cmodule PRIVATE cxx_std_14)
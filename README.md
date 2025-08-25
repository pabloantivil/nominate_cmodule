# DW-NOMINATE C++ Implementation

Una implementación moderna en C++ del algoritmo **DW-NOMINATE** (Dynamic Weighted NOMINAl Three-step Estimation) para el análisis espacial de datos de votaciones políticas. Basado en la implementación de DW-NOMINATE en python (https://github.com/pabloantivil/pynominate -> forked from https://github.com/voteview/pynominate)

## 📋 Descripción

DW-NOMINATE es un algoritmo desarrollado por Keith Poole y Howard Rosenthal para estimar las posiciones ideológicas de legisladores y la dificultad de propuestas legislativas en un espacio político multidimensional. Esta implementación utiliza técnicas modernas de optimización no lineal y álgebra lineal para proporcionar estimaciones precisas y eficientes.

## 🛠️ Tecnologías Utilizadas

- **C++14**: Lenguaje de programación principal
- **[Eigen](https://eigen.tuxfamily.org/)**: Librería de álgebra lineal para operaciones matriciales
- **[NLopt](https://nlopt.readthedocs.io/)**: Librería de optimización no lineal para estimación de parámetros
- **CMake**: Sistema de construcción multiplataforma
- **MinGW**: Compilador GCC para Windows

## 📁 Estructura del Proyecto

```
nominate_cmodule/
├── CMakeLists.txt          # Configuración de CMake
├── src/
│   └── main.cpp           # Archivo principal con pruebas
├── Eigen/                 # Librería Eigen (headers)
├── nlopt/                 # Librería NLopt
│   ├── include/
│   └── lib/
└── build/                 # Directorio de compilación
```

## 🚀 Instalación y Configuración

### Prerrequisitos

- **CMake** 3.15 o superior
- **MinGW** con GCC 6.3.0 o superior
- **Git** (para clonar el repositorio)

### Pasos de Instalación

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/[pabloantivil]/nominate_cmodule
   cd nominate_cmodule
   ```

2. **Compilar el proyecto:**
   ```bash
   mkdir build
   cd build
   cmake ..
   cmake --build .
   ```

## 🧮 Algoritmo DW-NOMINATE

### Características Principales

- **Análisis espacial**: Mapeo de votaciones en espacios ideológicos multidimensionales
- **Estimación bayesiana**: Utiliza métodos de máxima verosimilitud para parámetros
- **Optimización robusta**: Implementa algoritmos L-BFGS para convergencia eficiente
- **Escalabilidad**: Manejo eficiente de grandes conjuntos de datos de votaciones

### Modelo Matemático

El modelo estima:
- **Puntos ideales** de legisladores en el espacio político
- **Parámetros de propuestas** (punto medio y normal al plano de corte)
- **Probabilidades de voto** basadas en la utilidad espacial

## 📊 Uso

```cpp
#include <iostream>
#include <Eigen/Dense>
#include <nlopt.h>

// Ejemplo básico de optimización con NLopt
int main() {
    // Configurar problema de optimización
    nlopt_opt opt = nlopt_create(NLOPT_LD_LBFGS, 2);
    
    // Implementar función objetivo DW-NOMINATE
    nlopt_set_min_objective(opt, nominate_objective, nullptr);
    
    // Ejecutar optimización
    double x[2] = {0.0, 0.0};
    double minf;
    nlopt_optimize(opt, x, &minf);
    
    // Resultados
    std::cout << "Posición estimada: (" << x[0] << ", " << x[1] << ")" << std::endl;
    
    nlopt_destroy(opt);
    return 0;
}
```

## 🔬 Casos de Uso

- **Análisis de congresos**: Mapeo ideológico de legisladores
- **Estudios comparativos**: Análisis temporal de polarización política
- **Investigación académica**: Estudios de comportamiento legislativo
- **Consultorías políticas**: Análisis estratégico de votaciones

## 📚 Referencias

- Poole, Keith T., and Howard Rosenthal. *Congress: A Political-Economic History of Roll Call Voting*. Oxford University Press, 1997.
- Lewis, Jeffrey B., et al. "Voteview: Congressional roll-call votes database." (2021).
- Clinton, Joshua, Simon Jackman, and Douglas Rivers. "The statistical analysis of roll call data." *American Political Science Review* 98.2 (2004): 355-370.

## 👨‍💻 Autor

**[Pablo Antivil]**
- GitHub: [@pabloantivil](https://github.com/pabloantivil)
- Email: p.antivilmorales@gmail.com

## 🙏 Agradecimientos

- Keith Poole y Howard Rosenthal por el desarrollo original del algoritmo DW-NOMINATE
- Comunidad de Eigen por la excelente librería de álgebra lineal
- Desarrolladores de NLopt por las herramientas de optimización

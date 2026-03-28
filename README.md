# DW-NOMINATE C++

Implementación en C++ del algoritmo **DW-NOMINATE** (Dynamic Weighted NOMINAl Three-step Estimation) para estimar coordenadas ideológicas a partir de datos de votaciones legislativas.

## Descripción

Este proyecto reimplementa el algoritmo DW-NOMINATE originalmente desarrollado en R por Keith Poole y Howard Rosenthal. La implementación en C++ ofrece mejor rendimiento computacional y ha sido validada comparando sus resultados con los outputs generados por el paquete oficial de R (`dwnominate`).

El sistema procesa matrices de votaciones legislativas y estima las posiciones ideológicas de los legisladores en un espacio multidimensional, permitiendo analizar la evolución temporal de estas posiciones.

## Estructura del Repositorio

```
nominate_cmodule/
├── src/                    # Código fuente C++
├── include/                # Headers
├── Eigen/                  # Biblioteca de álgebra lineal
├── input_R/                # Datos de votaciones (entrada)
├── output_wnominate/       # Coordenadas iniciales W-NOMINATE
├── output_R_dwnominate_*/  # Parámetros de referencia por modelo
├── output_cpp/             # Resultados generados (salida)
├── tests/                  # Tests de validación
├── build/                  # Directorio de compilación
└── CMakeLists.txt
```

## Requisitos del Sistema

- **Compilador**: GCC 9+ o compatible con C++17
- **CMake**: 3.15+
- **Dependencias**:
  - OpenMP (paralelización)
  - NLopt (optimización numérica)
  - OpenBLAS (opcional, acelera álgebra lineal)

### Instalación de dependencias (MSYS2/MinGW64)

```bash
pacman -S mingw-w64-x86_64-cmake mingw-w64-x86_64-nlopt mingw-w64-x86_64-openblas
```

## Compilación

```bash
mkdir build && cd build
cmake .. -G "MinGW Makefiles"
mingw32-make -j4
```

El ejecutable `dwnominate.exe` se genera en el directorio `build/`.

## Uso

```bash
./dwnominate [opciones]
```

### Opciones

| Opción | Descripción | Default |
|--------|-------------|---------|
| `--model=<0\|1\|2\|3>` | Modelo temporal: 0=constante, 1=lineal, 2=cuadrático, 3=cúbico | 1 |
| `--iterations=<n>` | Número de iteraciones | 4 |
| `--periods=<n>` | Número de períodos legislativos | auto-detectar |
| `--dimensions=<n>` | Dimensiones espaciales | 2 |
| `--input-dir=<path>` | Directorio de votaciones | input_R |
| `--output-dir=<path>` | Directorio de salida | output_cpp |
| `--verbose` | Mostrar progreso detallado | - |
| `--help` | Mostrar ayuda | - |

### Ejemplos

```bash
# Ejecutar con modelo lineal y 10 iteraciones
./dwnominate --model=1 --iterations=10 --verbose

# Ejecutar con modelo constante
./dwnominate --model=0 --iterations=4

# Especificar directorios personalizados
./dwnominate --input-dir=mis_datos --output-dir=resultados --periods=5
```

## Inputs Requeridos

| Archivo | Ubicación | Descripción |
|---------|-----------|-------------|
| Matrices de votación | `input_R/votes_matrix_p*.csv` | Votaciones por período (1=Sí, 6=No, 9=Ausente) |
| Coordenadas iniciales | `output_wnominate/wnominate_coordinates.csv` | Estimaciones W-NOMINATE como punto de partida |
| Parámetros de bills | `output_R_dwnominate_model*/dwnominate_bill_parameters.csv` | Cutting points y spreads de referencia |

## Outputs Generados

Los resultados se exportan a `output_cpp/`:

| Archivo | Contenido |
|---------|-----------|
| `cpp_coordinates_all_periods.csv` | Coordenadas ideológicas por legislador y período |
| `cpp_bill_parameters.csv` | Parámetros estimados de cada votación |
| `cpp_summary.csv` | Estadísticas globales del modelo |

## Validación

El proyecto incluye un validador que compara los resultados de C++ contra los outputs de R:

```bash
./test_validate_csv
```

## Referencias

- Poole, K. T., & Rosenthal, H. (1985). A Spatial Model for Legislative Roll Call Analysis. *American Journal of Political Science*.
- Poole, K. T. (2005). *Spatial Models of Parliamentary Voting*. Cambridge University Press.
- [Paquete dwnominate en R](https://github.com/wmay/dwnominate)

## Licencia

Este proyecto está licenciado bajo la **MIT License** — consulta el archivo [LICENSE](LICENSE) para más detalles.

© 2026 Pablo Antivil. Si usas este código en tu investigación o proyecto, por favor da crédito al autor original.

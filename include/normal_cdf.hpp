#ifndef NORMAL_CDF_HPP
#define NORMAL_CDF_HPP

#include <vector>
#include <cmath>
#include <stdexcept>

/**
 * @brief Tabla de búsqueda precalculada para CDF normal estándar y valores relacionados.
 *
 * Esta clase implementa la funcionalidad de la subrutina Fortran init_zdf(),
 * que precalcula una tabla de búsqueda para la función de distribución acumulada
 * (CDF) normal estándar para acelerar evaluaciones repetidas durante la optimización
 * DW-NOMINATE.
 *
 * La tabla cubre el rango [-5.0, 5.0] con una resolución de 0.0001 (10,000 puntos por unidad).
 *
 * Equivalente en Fortran: init_zdf() en el módulo zdf_mod (líneas 39-63)
 * Tabla Fortran: ZDF(150000, 4) con columnas:
 *   Columna 1: valor z
 *   Columna 2: CDF(z)
 *   Columna 3: log(CDF(z))
 *   Columna 4: pdf(z) / CDF(z)
 */
class NormalCDF
{
public:
    /**
     * @brief Constructor que precalcula la tabla de búsqueda CDF.
     *
     * Equivalente a llamar init_zdf() una vez en Fortran.
     */
    NormalCDF();

    /**
     * @brief Obtener el valor CDF para un puntaje z dado.
     *
     * @param z Puntaje z normal estándar
     * @return CDF(z) = P(Z <= z) where Z ~ N(0,1)
     *
     * Utiliza interpolación lineal en la tabla precalculada.
     */
    double cdf(double z) const;

    /**
     * @brief Obtener el logaritmo natural del valor CDF.
     *
     * @param z Puntaje z normal estándar
     * @return log(CDF(z))
     *
     * Más numéricamente estable para probabilidades muy pequeñas que log(cdf(z)).
     */
    double logCdf(double z) const;

    /**
     * @brief Obtener la razón pdf(z)/CDF(z).
     *
     * @param z Puntaje z normal estándar
     * @return pdf(z) / CDF(z)
     *
     * Usado en cálculos de derivadas en PROX y PROLLC2.
     */
    double pdfOverCdf(double z) const;

    /**
     * @brief Obtener el valor z crudo de la tabla en el índice dado.
     *
     * @param index Índice de la tabla (basado en 0, de 0 a tableSize-1)
     * @return Valor z
     *
     * Para propósitos de validación y pruebas.
     */
    double getZ(size_t index) const;

    /**
     * @brief Obtener el tamaño de la tabla.
     *
     * @return Número de filas en la tabla de búsqueda
     */
    size_t getTableSize() const { return tableSize_; }

    /**
     * @brief Obtener la resolución (puntos por unidad).
     *
     * @return Valor de resolución (10000.0)
     */
    double getResolution() const { return resolution_; }

private:
    /**
     * @brief Inicializar la tabla de búsqueda (llamado por el constructor).
     *
     * Traducción directa de la lógica init_zdf() de Fortran.
     */
    void initializeTable();

    /**
     * @brief Realizar interpolación lineal en la tabla.
     *
     * @param z Puntaje z a buscar
     * @param column Índice de columna (0=z, 1=CDF, 2=logCDF, 3=pdf/CDF)
     * @return Valor interpolado
     */
    double interpolate(double z, int column) const;

    // Constantes de Fortran
    static constexpr size_t NDEVIT = 50001;      // Puntos por lado (positivo/negativo)
    static constexpr double XDEVIT = 10000.0;    // Factor de resolución
    static constexpr double PI = 3.1415926536;   // Misma precisión que Fortran
    static constexpr size_t TABLE_ROWS = 150000; // Total de filas en la tabla ZDF (dimensión Fortran)

    // Estructura de la tabla: equivalente a ZDF(150000, 4) en Fortran
    // C++ usa almacenamiento en filas principales: table_[row * 4 + column]
    std::vector<double> table_; // Arreglo 2D aplanado
    size_t tableSize_;          // Filas realmente usadas: 2*NDEVIT - 1 = 100001
    double resolution_;         // XDEVIT
    double minZ_;               // Valor mínimo z en la tabla (~-5.0)
    double maxZ_;               // Valor máximo z en la tabla (~5.0)
};

#endif // NORMAL_CDF_HPP

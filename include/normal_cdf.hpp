#ifndef NORMAL_CDF_HPP
#define NORMAL_CDF_HPP

#include <vector>
#include <cmath>
#include <stdexcept>
#include <utility> // Para std::pair

/**
 * Tabla de búsqueda precalculada para CDF normal estándar y valores relacionados.
 * La tabla cubre el rango [-5.0, 5.0] con una resolución de 0.0001 (10,000 puntos por unidad).
 */
class NormalCDF
{
public:
    // Constructor que precalcula la tabla de búsqueda CDF.
    NormalCDF();

    /**
     * Obtener el valor CDF para un puntaje z dado.
     * @param z Puntaje z normal estándar
     * @return CDF(z) = P(Z <= z) where Z ~ N(0,1)
     * Utiliza interpolación lineal en la tabla precalculada.
     */
    double cdf(double z) const;

    /**
     * Obtener el logaritmo natural del valor CDF.
     * @param z Puntaje z normal estándar
     * @return log(CDF(z))
     * Más numéricamente estable para probabilidades muy pequeñas que log(cdf(z)).
     */
    double logCdf(double z) const;

    /**
     * Obtener la razón pdf(z)/CDF(z).
     * @param z Puntaje z normal estándar
     * @return pdf(z) / CDF(z)
     *
     */
    double pdfOverCdf(double z) const;

    /**
     * Obtener la razón exp(-z²/2)/CDF(z) - Compatible con Fortran.
     * @param z Puntaje z normal estándar
     * @return exp(-z²/2) / CDF(z) (sin el factor 1/sqrt(2*pi))
     * 
     * Esto es igual al cálculo ZGAUSS/ZDISTF en Fortran
     * donde ZGAUSS = exp(-(ZS*ZS)/2.0) y ZDISTF = CDF(ZS).
     */
    double gaussOverCdf(double z) const;

    /**
     * OPTIMIZADO: Obtener logCdf Y gaussOverCdf en una sola llamada.
     * @param z Puntaje z normal estándar
     * @return Par (logCdf, gaussOverCdf) calculados con una sola búsqueda
     * Evita duplicar la búsqueda lineal en la tabla.
     */
    std::pair<double, double> logCdfAndMills(double z) const;

    /**
     * Obtener el valor z crudo de la tabla en el índice dado.
     * @param index Índice de la tabla (basado en 0, de 0 a tableSize-1)
     * @return Valor z
     * Para propósitos de validación y pruebas.
     */
    double getZ(size_t index) const;

    /**
     * Obtener el tamaño de la tabla.
     * @return Número de filas en la tabla de búsqueda
     */
    size_t getTableSize() const { return tableSize_; }

    /**
     * Obtener la resolución (puntos por unidad).
     * @return Valor de resolución (10000.0)
     */
    double getResolution() const { return resolution_; }

private:
    /**
     * Inicializar la tabla de búsqueda (llamado por el constructor).
     * Traducción directa de la lógica init_zdf() de Fortran.
     */
    void initializeTable();

    /**
     * Realizar interpolación lineal en la tabla.
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
    std::vector<double> table_; // Arreglo 2D aplanado
    size_t tableSize_;          // Filas realmente usadas: 2*NDEVIT - 1 = 100001
    double resolution_;         // XDEVIT
    double minZ_;               // Valor mínimo z en la tabla (~-5.0)
    double maxZ_;               // Valor máximo z en la tabla (~5.0)
};

#endif // NORMAL_CDF_HPP

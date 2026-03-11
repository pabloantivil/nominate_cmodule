#include "normal_cdf.hpp"
#include <cmath>
#include <algorithm>

NormalCDF::NormalCDF()
    : tableSize_(2 * NDEVIT - 1), resolution_(XDEVIT), table_(TABLE_ROWS * 4, 0.0),
      minZ_(0.0), maxZ_(0.0)
{
    initializeTable();

    // Guardar min/max valores z para la verificación de límites
    minZ_ = table_[0];                    // Primer valor z (más negativo)
    maxZ_ = table_[(tableSize_ - 1) * 4]; // Último valor z (más positivo)
}

void NormalCDF::initializeTable()
{
    // Matrices temporales equivalentes a YY y CUMNML en Fortran
    std::vector<double> yy(NDEVIT);
    std::vector<double> cumnml(NDEVIT);

    for (size_t i = 0; i < NDEVIT; ++i)
    {
        yy[i] = static_cast<double>(i) / XDEVIT;

        double x = yy[i] / std::sqrt(2.0);
        double xx = std::erf(x); // std::erf() es equivalente a Fortran ERF()
        xx = xx / 2.0 + 0.5;
        cumnml[i] = xx;
    }

    const double twopi = 1.0 / std::sqrt(2.0 * PI);

    for (size_t i = 0; i < NDEVIT; ++i)
    {
        size_t fortranIndex = NDEVIT - 1 - i; // Mapea a Fortran NDEVIT+1-I con base 0

        // Columna 1 (índice 0): valor z (lado negativo)
        table_[i * 4 + 0] = yy[fortranIndex] * (-1.0);

        // Columna 2 (índice 1): valor CDF
        table_[i * 4 + 1] = 1.0 - cumnml[fortranIndex];

        // Columna 3 (índice 2): log(CDF)
        table_[i * 4 + 2] = std::log(table_[i * 4 + 1]);
    }

    // FORTRAN LOOP 902: Llenar el lado positivo de la tabla
    for (size_t i = 1; i < NDEVIT; ++i)
    { // Comienza en 1, no en 0 (Fortran comienza en 2)

        size_t rowIndex = i - 1 + NDEVIT;

        // Columna 1 (índice 0): valor z (lado positivo)
        table_[rowIndex * 4 + 0] = yy[i];

        // Columna 2 (índice 1): valor CDF
        table_[rowIndex * 4 + 1] = cumnml[i];

        // Columna 3 (índice 2): log(CDF)
        table_[rowIndex * 4 + 2] = std::log(table_[rowIndex * 4 + 1]);
    }

    // FORTRAN LOOP 903: Calcular la razón pdf/CDF (columna 4)
    for (size_t i = 0; i < tableSize_; ++i)
    {
        double z = table_[i * 4 + 0];
        double cdf = table_[i * 4 + 1];

        // pdf(z) = (1/sqrt(2*pi)) * exp(-z^2/2)
        // Columna 4 (índice 3): pdf(z) / CDF(z)
        table_[i * 4 + 3] = (twopi * std::exp((-z * z) / 2.0)) / cdf;
    }
}

double NormalCDF::interpolate(double z, int column) const
{
    // Manejar valores fuera de límites
    if (z <= minZ_)
    {
        return table_[0 * 4 + column];
    }
    if (z >= maxZ_)
    {
        return table_[(tableSize_ - 1) * 4 + column];
    }

    // OPTIMIZADO: Cálculo directo de índice O(1) en lugar de búsqueda lineal O(n)
    // La tabla tiene valores z desde minZ_ (-5.0) hasta maxZ_ (+5.0)
    // con espaciado de 1/resolution_ (0.0001)
    double indexFloat = (z - minZ_) * resolution_;
    size_t lowerIndex = static_cast<size_t>(indexFloat);

    // Asegurar que no excedemos los límites
    if (lowerIndex >= tableSize_ - 1)
    {
        lowerIndex = tableSize_ - 2;
    }
    size_t upperIndex = lowerIndex + 1;

    // Obtener los valores z y los valores de la columna objetivo
    double z_lower = table_[lowerIndex * 4 + 0];
    double z_upper = table_[upperIndex * 4 + 0];
    double value_lower = table_[lowerIndex * 4 + column];
    double value_upper = table_[upperIndex * 4 + column];

    // Interpolación lineal
    // Evitar división por cero
    if (std::abs(z_upper - z_lower) < 1e-10)
    {
        return value_lower;
    }

    double t = (z - z_lower) / (z_upper - z_lower);
    return value_lower + t * (value_upper - value_lower);
}

double NormalCDF::cdf(double z) const
{
    return interpolate(z, 1); // Columna 2 en Fortran (índice 1 en C++)
}

double NormalCDF::logCdf(double z) const
{
    return interpolate(z, 2); // Columna 3 en Fortran (índice 2 en C++)
}

double NormalCDF::pdfOverCdf(double z) const
{
    return interpolate(z, 3); // Columna 4 en Fortran (índice 3 en C++)
}

double NormalCDF::gaussOverCdf(double z) const
{
    // Fortran-compatible: ZGAUSS/ZDISTF where ZGAUSS = exp(-ZS²/2)
    // This is the Mills ratio without the 1/sqrt(2*pi) factor
    double cdf_val = cdf(z);
    if (cdf_val < 1e-300)
    {
        cdf_val = 1e-300; // Avoid division by zero
    }
    return std::exp(-z * z / 2.0) / cdf_val;
}

std::pair<double, double> NormalCDF::logCdfAndMills(double z) const
{
    // OPTIMIZADO: Cálculo directo de índice O(1) + una sola búsqueda para ambos valores
    // Manejar valores fuera de límites
    if (z <= minZ_)
    {
        double logCdfVal = table_[0 * 4 + 2];
        double cdfVal = table_[0 * 4 + 1];
        if (cdfVal < 1e-300)
            cdfVal = 1e-300;
        double millsVal = std::exp(-z * z / 2.0) / cdfVal;
        return {logCdfVal, millsVal};
    }
    if (z >= maxZ_)
    {
        double logCdfVal = table_[(tableSize_ - 1) * 4 + 2];
        double cdfVal = table_[(tableSize_ - 1) * 4 + 1];
        if (cdfVal < 1e-300)
            cdfVal = 1e-300;
        double millsVal = std::exp(-z * z / 2.0) / cdfVal;
        return {logCdfVal, millsVal};
    }

    // OPTIMIZADO: Cálculo directo de índice O(1)
    double indexFloat = (z - minZ_) * resolution_;
    size_t lowerIndex = static_cast<size_t>(indexFloat);
    if (lowerIndex >= tableSize_ - 1)
    {
        lowerIndex = tableSize_ - 2;
    }
    size_t upperIndex = lowerIndex + 1;

    // Obtener los valores z
    double z_lower = table_[lowerIndex * 4 + 0];
    double z_upper = table_[upperIndex * 4 + 0];

    // Calcular factor de interpolación
    double t = 0.0;
    if (std::abs(z_upper - z_lower) >= 1e-10)
    {
        t = (z - z_lower) / (z_upper - z_lower);
    }

    // Interpolar logCdf (columna 2)
    double logCdf_lower = table_[lowerIndex * 4 + 2];
    double logCdf_upper = table_[upperIndex * 4 + 2];
    double logCdfVal = logCdf_lower + t * (logCdf_upper - logCdf_lower);

    // Interpolar CDF (columna 1) para calcular Mills ratio
    double cdf_lower = table_[lowerIndex * 4 + 1];
    double cdf_upper = table_[upperIndex * 4 + 1];
    double cdfVal = cdf_lower + t * (cdf_upper - cdf_lower);
    if (cdfVal < 1e-300)
        cdfVal = 1e-300;
    double millsVal = std::exp(-z * z / 2.0) / cdfVal;

    return {logCdfVal, millsVal};
}

double NormalCDF::getZ(size_t index) const
{
    if (index >= tableSize_)
    {
        throw std::out_of_range("Index out of range in NormalCDF table");
    }
    return table_[index * 4 + 0]; // Columna 1 en Fortran (índice 0 en C++)
}

#ifndef SORT_UTILS_HPP
#define SORT_UTILS_HPP
#include <vector>
#include <algorithm>
#include <numeric>
#include <Eigen/Dense>

/**
 * Utilidades de ordenamiento para DW-NOMINATE.
 * RSORT implementa ordenamiento hibrido Quicksort + Insertion Sort que ordena valores
 * en orden ascendente mientras mantiene sincronizado un vector de indices
 * Se usa std::sort con el patron argsort para simplicidad
 */

/**
 * Ordena valores y retorna indices de la permutacion (argsort)
 * @param values Vector de valores a ordenar (no se modifica)
 * @return Vector de indices ordenados [0, n-1] tal que values[indices[i]] esta ordenado
 */
template <typename T>
std::vector<size_t> argsort(const std::vector<T> &values)
{
    const size_t n = values.size();
    std::vector<size_t> indices(n);

    // Inicializar indices: [0, 1, 2, ..., n-1]
    std::iota(indices.begin(), indices.end(), 0);

    // Ordenar indices segun los valores (orden ascendente)
    std::sort(indices.begin(), indices.end(),
              [&values](size_t a, size_t b)
              { return values[a] < values[b]; });

    return indices;
}

/**
 * Ordena valores de Eigen::VectorXd y retorna indices.
 * Sobrecarga para vectores de Eigen.
 * @param values Eigen::VectorXd con valores a ordenar
 * @return Vector de indices ordenados
 */
inline std::vector<size_t> argsort(const Eigen::VectorXd &values)
{
    const size_t n = values.size();
    std::vector<size_t> indices(n);

    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(),
              [&values](size_t a, size_t b)
              { return values[a] < values[b]; });

    return indices;
}

/**
 * Ordena valores en orden DESCENDENTE y retorna indices.
 * @param values Vector de valores a ordenar
 * @return Vector de indices en orden descendente
 */
template <typename T>
std::vector<size_t> argsort_descending(const std::vector<T> &values)
{
    const size_t n = values.size();
    std::vector<size_t> indices(n);

    std::iota(indices.begin(), indices.end(), 0);

    // Orden descendente: mayor primero
    std::sort(indices.begin(), indices.end(),
              [&values](size_t a, size_t b)
              { return values[a] > values[b]; });

    return indices;
}

// Ordena valores de Eigen en orden descendente.
inline std::vector<size_t> argsort_descending(const Eigen::VectorXd &values)
{
    const size_t n = values.size();
    std::vector<size_t> indices(n);

    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(),
              [&values](size_t a, size_t b)
              { return values[a] > values[b]; });

    return indices;
}

/**
 * Aplica una permutacion a un vector usando indices.
 * Reordena los elementos de un vector segun la permutacion especificada.
 * Util si se necesita modificar el vector original
 * @param values Vector a reordenar (se modifica)
 * @param indices Permutacion a aplicar
 */
template <typename T>
void applyPermutation(std::vector<T> &values, const std::vector<size_t> &indices)
{
    std::vector<T> temp(values.size());
    for (size_t i = 0; i < indices.size(); ++i)
    {
        temp[i] = values[indices[i]];
    }
    values = std::move(temp);
}

// Aplica permutacion a Eigen::VectorXd.
inline void applyPermutation(Eigen::VectorXd &values, const std::vector<size_t> &indices)
{
    Eigen::VectorXd temp(values.size());
    for (size_t i = 0; i < indices.size(); ++i)
    {
        temp[i] = values[indices[i]];
    }
    values = temp;
}

/**
 * Interfaz compatible con RSORT de Fortran (modifica in-place).
 * @param values Vector de valores (se modifica in-place)
 * @param indices Vector de indices (se modifica, 1-based)
 */
inline void rsort_fortran_style(std::vector<double> &values, std::vector<int> &indices)
{
    const size_t n = values.size();

    // Crear indices 0-based temporales
    std::vector<size_t> indices0(n);
    for (size_t i = 0; i < n; ++i)
    {
        indices0[i] = indices[i] - 1; // Convertir de 1-based a 0-based
    }

    // Ordenar indices segun valores
    std::sort(indices0.begin(), indices0.end(),
              [&values](size_t a, size_t b)
              { return values[a] < values[b]; });

    // Aplicar permutacion a values e indices
    std::vector<double> sortedValues(n);
    std::vector<int> sortedIndices(n);

    for (size_t i = 0; i < n; ++i)
    {
        sortedValues[i] = values[indices0[i]];
        sortedIndices[i] = static_cast<int>(indices0[i]) + 1; // Convertir a 1-based
    }

    values = std::move(sortedValues);
    indices = std::move(sortedIndices);
}

#endif // SORT_UTILS_HPP

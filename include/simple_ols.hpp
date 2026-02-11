#ifndef SIMPLE_OLS_HPP
#define SIMPLE_OLS_HPP

/**
 * @file simple_ols.hpp
 * @brief Subrutina REGA (OLS) a C++.
 *
 * Implementa la regresion OLS clasica usando la formula:
 *   beta = (X'X)^-1 X' y
 *
 * La inversion de X'X se realiza mediante eigendescomposicion,
 * replicando el comportamiento de REGA (pseudo-inversa con umbral 0.01).
 */

#include <Eigen/Dense>

/**
 * @brief Regresion OLS simple
 *
 * @param ns Numero de observaciones (filas de A)
 * @param nf Numero de variables (columnas de A)
 * @param A  Matriz de diseno (al menos ns x nf)
 * @param y  Vector de respuestas (al menos ns)
 * @param eigenThreshold Umbral para eigenvalores (por defecto 0.01)
 * @return Vector de coeficientes beta (tamano nf)
 */
Eigen::VectorXd simpleOLS(
    int ns,
    int nf,
    const Eigen::MatrixXd &A,
    const Eigen::VectorXd &y,
    double eigenThreshold = 0.01);

#endif // SIMPLE_OLS_HPP

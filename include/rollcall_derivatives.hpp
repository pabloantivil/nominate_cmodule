#ifndef ROLLCALL_DERIVATIVES_HPP
#define ROLLCALL_DERIVATIVES_HPP

/**
 * Calculo de derivadas y log-likelihood para parametros de roll calls
 * PROLLC2 calcula derivadas del log-likelihood respecto a los parametros
 * de una votacion especifica (midpoint y spread), necesarias para el
 * optimizador RCINT2
 */

#include "likelihood.hpp"
#include "normal_cdf.hpp"
#include <Eigen/Dense>
#include <vector>
#include <array>

// Buffer de trabajo para computeRollCallDerivatives.
struct RollCallDerivativesWorkBuffer
{
    // Arrays fijos + pesos cacheados
    std::array<double, MAX_DIMENSIONS> weightsSquared;
    std::array<double, MAX_DIMENSIONS> midpointDeriv;
    std::array<double, MAX_DIMENSIONS> spreadDeriv;
    int numDimensions;

    RollCallDerivativesWorkBuffer() : numDimensions(0)
    {
        weightsSquared.fill(0.0);
        midpointDeriv.fill(0.0);
        spreadDeriv.fill(0.0);
    }

    void cacheWeights(const Eigen::VectorXd &weights, int ns)
    {
        numDimensions = ns;
        for (int k = 0; k < ns && k < MAX_DIMENSIONS; ++k)
        {
            weightsSquared[k] = weights(k) * weights(k);
        }
    }

    void reset()
    {
        for (int k = 0; k < numDimensions && k < MAX_DIMENSIONS; ++k)
        {
            midpointDeriv[k] = 0.0;
            spreadDeriv[k] = 0.0;
        }
    }
};

// Resultado del calculo de derivadas para un roll call.
struct RollCallDerivativesResult
{
    // Log-likelihood
    double logLikelihood;     // XPLOG: suma de log(CDF) para este roll call
    double geometricMeanProb; // GMP: exp(XPLOG/KTOT)

    // Derivadas (vectores de dimension NS)
    Eigen::VectorXd midpointDerivatives; // ZDERV: dL/d(zmid_k)
    Eigen::VectorXd spreadDerivatives;   // DDERV: dL/d(dyn_k)

    // Estadisticas de clasificacion
    int totalVotes;        // KTOT: votos validos procesados
    int correctClassified; // KLASS: clasificaciones correctas (|DC| <= |DB|)

    // Matriz de confusion detallada
    int truePositives;  // KLASSYY: predijo Si correctamente
    int falseNegatives; // KLASSNY: predijo No, era Si
    int falsePositives; // KLASSYN: predijo Si, era No
    int trueNegatives;  // KLASSNN: predijo No correctamente

    // Clasificacion alternativa
    int positiveZS; // KLASS2: votos con ZS > 0

    // Constructor por defecto
    RollCallDerivativesResult()
        : logLikelihood(0.0),
          geometricMeanProb(0.0),
          totalVotes(0),
          correctClassified(0),
          truePositives(0),
          falseNegatives(0),
          falsePositives(0),
          trueNegatives(0),
          positiveZS(0)
    {
    }

    // Constructor con dimension especificada.
    explicit RollCallDerivativesResult(int numDimensions)
        : logLikelihood(0.0),
          geometricMeanProb(0.0),
          midpointDerivatives(Eigen::VectorXd::Zero(numDimensions)),
          spreadDerivatives(Eigen::VectorXd::Zero(numDimensions)),
          totalVotes(0),
          correctClassified(0),
          truePositives(0),
          falseNegatives(0),
          falsePositives(0),
          trueNegatives(0),
          positiveZS(0)
    {
    }

    // Calcula precision de clasificacion.
    double getAccuracy() const
    {
        return totalVotes > 0
                   ? static_cast<double>(correctClassified) / totalVotes
                   : 0.0;
    }
};

/**
 * Calcula log-likelihood y derivadas para un roll call especifico.
 * @param legislatorCoords Coordenadas de legisladores (numLeg x numDim)
 * @param rollCallIndex Indice del roll call a evaluar (0-based)
 * @param midpoint Punto medio actual del roll call (zmid)
 * @param spread Spread actual del roll call (dyn)
 * @param votes Matriz de votos
 * @param weights Vector de pesos [w1, w2, ..., wNS, beta]
 * @param normalCDF Tabla CDF precomputada
 * @return Resultado con log-likelihood, derivadas y estadisticas
 */
RollCallDerivativesResult computeRollCallDerivatives(
    const Eigen::MatrixXd &legislatorCoords,
    int rollCallIndex,
    const Eigen::VectorXd &midpoint,
    const Eigen::VectorXd &spread,
    const VoteMatrix &votes,
    const Eigen::VectorXd &weights,
    const NormalCDF &normalCDF);

// Version simplificada que usa RollCallParameters.
RollCallDerivativesResult computeRollCallDerivatives(
    const Eigen::MatrixXd &legislatorCoords,
    int rollCallIndex,
    const RollCallParameters &rcParams,
    const VoteMatrix &votes,
    const Eigen::VectorXd &weights,
    const NormalCDF &normalCDF);

/**
 * Version optimizada de computeRollCallDerivatives.
 * OPTIMIZACIONES:
 * - Arrays de tamano fijo (sin heap allocations)
 * - Buffer de trabajo reutilizable
 * - Pesos al cuadrado pre-cacheados en buffer
 * - Filtrado por legisladores validos (opcional)
 * - Calculos inline de distancia y utilidad
 */
RollCallDerivativesResult computeRollCallDerivativesOptimized(
    const Eigen::MatrixXd &legislatorCoords,
    int rollCallIndex,
    const Eigen::VectorXd &midpoint,
    const Eigen::VectorXd &spread,
    const VoteMatrix &votes,
    const Eigen::VectorXd &weights,
    const NormalCDF &normalCDF,
    RollCallDerivativesWorkBuffer &buffer);

// Version optimizada con lista de legisladores validos.
RollCallDerivativesResult computeRollCallDerivativesOptimized(
    const Eigen::MatrixXd &legislatorCoords,
    int rollCallIndex,
    const Eigen::VectorXd &midpoint,
    const Eigen::VectorXd &spread,
    const VoteMatrix &votes,
    const Eigen::VectorXd &weights,
    const NormalCDF &normalCDF,
    RollCallDerivativesWorkBuffer &buffer,
    const std::vector<int> &validLegislators);

#endif // ROLLCALL_DERIVATIVES_HPP

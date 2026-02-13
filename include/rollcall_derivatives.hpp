#ifndef ROLLCALL_DERIVATIVES_HPP
#define ROLLCALL_DERIVATIVES_HPP

/**
 * Calculo de derivadas y log-likelihood para parametros de roll calls
 *
 * PROLLC2 calcula derivadas del log-likelihood respecto a los parametros
 * de una votacion especifica (midpoint y spread), necesarias para el
 * optimizador RCINT2
 */

#include "likelihood.hpp"
#include "normal_cdf.hpp"
#include <Eigen/Dense>
#include <vector>

/**
 * Resultado del calculo de derivadas para un roll call.
 */
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

    /**
     * Constructor por defecto.
     */
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

    /**
     * Constructor con dimension especificada.
     */
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

    /**
     * Calcula precision de clasificacion.
     */
    double getAccuracy() const
    {
        return totalVotes > 0
                   ? static_cast<double>(correctClassified) / totalVotes
                   : 0.0;
    }
};

/**
 * Calcula log-likelihood y derivadas para un roll call especifico.
 *
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

/**
 * Version simplificada que usa RollCallParameters.
 */
RollCallDerivativesResult computeRollCallDerivatives(
    const Eigen::MatrixXd &legislatorCoords,
    int rollCallIndex,
    const RollCallParameters &rcParams,
    const VoteMatrix &votes,
    const Eigen::VectorXd &weights,
    const NormalCDF &normalCDF);

#endif // ROLLCALL_DERIVATIVES_HPP

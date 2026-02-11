#ifndef OPTIMIZE_LEGISLATORS_HPP
#define OPTIMIZE_LEGISLATORS_HPP

/**
 * @file optimize_legislators.hpp
 * @brief Optimizador de parametros de legisladores (XINT).
 *
 */

#include "legislator_derivatives.hpp"
#include "normal_cdf.hpp"
#include "simple_ols.hpp"
#include "sort_utils.hpp"
#include <Eigen/Dense>
#include <vector>

/**
 * @brief Configuracion del optimizador de legisladores.
 */
struct LegislatorOptimizerConfig
{
    int maxIterations = 10;           // Numero maximo de iteraciones por termino
    int numSearchPointsConst = 25;    // NINC para termino constante
    int numSearchPointsTemporal = 10; // NINC para terminos temporales
    double stepUnit = 0.01;           // Unidad base del paso
    double convergenceTol = 0.0001;   // Tolerancia de convergencia en GMP
    double gmpDropTol = 0.00001;      // Tolerancia para detectar empeoramiento
    double unitSphereScale = 0.75;    // Escala al proyectar a hiperesfera
    double eigenThreshold = 0.0001;   // Umbral de eigenvalores para inversion
};

/**
 * @brief Resultado de la optimizacion de un legislador.
 */
struct LegislatorOptimizationResult
{
    TemporalCoefficients coefficients; // XBETA optimizado

    double logLikelihood0 = 0.0; // XPLOG0
    double logLikelihood1 = 0.0; // XPLOG1
    double logLikelihood2 = 0.0; // XPLOG2
    double logLikelihood3 = 0.0; // XPLOG3

    int totalVotes = 0; // KXTOT

    Eigen::MatrixXd covariance0; // OUTX0S (invertida)
    Eigen::MatrixXd covariance1; // OUTX1S (invertida)
    Eigen::MatrixXd covariance2; // OUTX2S (invertida)
    Eigen::MatrixXd covariance3; // OUTX3S (invertida)

    Eigen::MatrixXd dervish;           // DERVISH (4 x NS)
    Eigen::MatrixXd periodCoordinates; // XMARK

    explicit LegislatorOptimizationResult(int numDimensions)
        : coefficients(numDimensions),
          covariance0(Eigen::MatrixXd::Zero(numDimensions, numDimensions)),
          covariance1(Eigen::MatrixXd::Zero(2 * numDimensions, 2 * numDimensions)),
          covariance2(Eigen::MatrixXd::Zero(3 * numDimensions, 3 * numDimensions)),
          covariance3(Eigen::MatrixXd::Zero(4 * numDimensions, 4 * numDimensions)),
          dervish(Eigen::MatrixXd::Zero(4, numDimensions))
    {
    }
};

/**
 * @brief Optimiza las coordenadas de un legislador (XINT).
 *
 * @param legislatorIndex Indice del legislador (NEP, 0-based)
 * @param periodInfo Informacion de periodos servidos (LWHERE/KWHERE)
 * @param legislatorDataCoords Matriz XDATA (dataIndex x numDim)
 * @param rollCallMidpoints Matriz ZMID (rollCall x numDim)
 * @param rollCallSpreads Matriz DYN (rollCall x numDim)
 * @param votes Matriz de votos
 * @param validRollCalls Vector de roll calls validos (RCBAD)
 * @param weights Vector de pesos [w1..wNS, beta]
 * @param normalCDF Tabla CDF precomputada
 * @param maxModel Modelo temporal maximo permitido
 * @param firstPeriod Periodo inicial (0-based)
 * @param lastPeriod Periodo final (0-based)
 * @param config Configuracion del optimizador
 * @return Resultado de la optimizacion del legislador
 */
LegislatorOptimizationResult optimizeLegislator(
    int legislatorIndex,
    const LegislatorPeriodInfo &periodInfo,
    const Eigen::MatrixXd &legislatorDataCoords,
    const Eigen::MatrixXd &rollCallMidpoints,
    const Eigen::MatrixXd &rollCallSpreads,
    const VoteMatrix &votes,
    const std::vector<bool> &validRollCalls,
    const Eigen::VectorXd &weights,
    const NormalCDF &normalCDF,
    TemporalModel maxModel,
    int firstPeriod,
    int lastPeriod,
    const LegislatorOptimizerConfig &config = LegislatorOptimizerConfig());

#endif // OPTIMIZE_LEGISLATORS_HPP

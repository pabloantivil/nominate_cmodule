#ifndef OPTIMIZE_LEGISLATORS_HPP
#define OPTIMIZE_LEGISLATORS_HPP

/**
 * @file optimize_legislators.hpp
 * Optimizador de parametros de legisladores (XINT).
 */

#include "legislator_derivatives.hpp"
#include "normal_cdf.hpp"
#include "simple_ols.hpp"
#include "sort_utils.hpp"
#include <Eigen/Dense>
#include <vector>

/**
 * Configuracion del optimizador de legisladores.
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
 * Resultado de la optimizacion de un legislador.
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
 * Optimiza las coordenadas de un legislador (XINT).
 * 
 * @return Resultado de la optimizacion del legislador
 */
LegislatorOptimizationResult optimizeLegislator(
    int legislatorIndex, // Indice del legislador (NEP, 0-based)
    const LegislatorPeriodInfo &periodInfo, // Informacion de periodos servidos (LWHERE/KWHERE)
    const Eigen::MatrixXd &legislatorDataCoords, // Matriz XDATA (dataIndex x numDim)
    const Eigen::MatrixXd &rollCallMidpoints, // Matriz ZMID (rollCall x numDim)
    const Eigen::MatrixXd &rollCallSpreads, // Matriz DYN (rollCall x numDim)
    const VoteMatrix &votes, // Matriz de votos
    const std::vector<bool> &validRollCalls, // Vector de roll calls validos (RCBAD)
    const Eigen::VectorXd &weights, // Vector de pesos [w1..wNS, beta]
    const NormalCDF &normalCDF, // Tabla CDF precomputada
    TemporalModel maxModel, // Modelo temporal maximo permitido
    int firstPeriod, // Periodo inicial (0-based)
    int lastPeriod, // Periodo final (0-based)
    const LegislatorOptimizerConfig &config = LegislatorOptimizerConfig()); // Configuracion del optimizador

#endif // OPTIMIZE_LEGISLATORS_HPP

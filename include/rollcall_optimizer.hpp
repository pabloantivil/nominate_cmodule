/**
 * Optimizador de parametros de roll calls (RCINT2).
 *
 * RCINT2 optimiza los parametros de cada roll call (votacion):
 * - ZMID (midpoint): Posicion del centro del plano de corte
 * - DYN (spread): Parametro que controla la orientacion del plano
 */

#ifndef ROLLCALL_OPTIMIZER_HPP
#define ROLLCALL_OPTIMIZER_HPP

#include "rollcall_derivatives.hpp"
#include "sort_utils.hpp"
#include <Eigen/Dense>
#include <functional>
#include <vector>

/**
 * Configuracion del optimizador de roll calls.
 */
struct RollCallOptimizerConfig
{
    int numOuterIterations = 5;     // Loop JJJJ: ciclos DYN+ZMID
    int numInnerIterations = 10;    // Iteraciones por fase
    int numSearchPoints = 25;       // NINC: puntos en busqueda lineal
    double stepUnit = 0.01;         // Unidad base para tamanio de paso
    double convergenceTol = 0.0001; // Tolerancia para convergencia
    double minSpread = 0.001;       // Minimo valor absoluto de spread
    double defaultSpread = 0.03;    // Valor por defecto si spread es muy pequenio
    double derivativeTol = 0.0001;  // Tolerancia para derivadas nulas

    RollCallOptimizerConfig() = default;
};

/**
 * Resultado de la optimizacion de un roll call.
 */
struct RollCallOptimizationResult
{
    // Parametros optimizados
    Eigen::VectorXd midpoint; // OLDZ optimizado
    Eigen::VectorXd spread;   // OLDD optimizado

    // Metricas finales
    double logLikelihood;     // XPLOG final
    double geometricMeanProb; // GMP final
    double initialGMP;        // GMP inicial (para comparacion)

    // Estadisticas de convergencia
    int totalIterations;    // Iteraciones totales ejecutadas
    int spreadIterations;   // Iteraciones en fase spread
    int midpointIterations; // Iteraciones en fase midpoint
    bool converged;         // Si convergio antes del limite

    // Clasificacion
    int totalVotes;
    int correctClassified;

    RollCallOptimizationResult()
        : logLikelihood(0.0),
          geometricMeanProb(0.0),
          initialGMP(0.0),
          totalIterations(0),
          spreadIterations(0),
          midpointIterations(0),
          converged(false),
          totalVotes(0),
          correctClassified(0)
    {
    }

    explicit RollCallOptimizationResult(int numDim)
        : midpoint(Eigen::VectorXd::Zero(numDim)),
          spread(Eigen::VectorXd::Zero(numDim)),
          logLikelihood(0.0),
          geometricMeanProb(0.0),
          initialGMP(0.0),
          totalIterations(0),
          spreadIterations(0),
          midpointIterations(0),
          converged(false),
          totalVotes(0),
          correctClassified(0)
    {
    }

    double getAccuracy() const
    {
        return totalVotes > 0
                   ? static_cast<double>(correctClassified) / totalVotes
                   : 0.0;
    }

    double getImprovement() const
    {
        return geometricMeanProb - initialGMP;
    }
};

/**
 * Optimiza los parametros de un roll call especifico.
 *
 * @param legislatorCoords Coordenadas de legisladores (numLeg x numDim)
 * @param rollCallIndex Indice del roll call a optimizar (0-based)
 * @param initialMidpoint Punto medio inicial (zmid)
 * @param initialSpread Spread inicial (dyn)
 * @param votes Matriz de votos
 * @param weights Vector de pesos [w1, ..., wNS, beta]
 * @param normalCDF Tabla CDF precomputada
 * @param config Configuracion del optimizador
 * @return Resultado con parametros optimizados y estadisticas
 */
RollCallOptimizationResult optimizeRollCall(
    const Eigen::MatrixXd &legislatorCoords,
    int rollCallIndex,
    const Eigen::VectorXd &initialMidpoint,
    const Eigen::VectorXd &initialSpread,
    const VoteMatrix &votes,
    const Eigen::VectorXd &weights,
    const NormalCDF &normalCDF,
    const RollCallOptimizerConfig &config = RollCallOptimizerConfig());

/**
 * Version con RollCallParameters.
 */
RollCallOptimizationResult optimizeRollCall(
    const Eigen::MatrixXd &legislatorCoords,
    int rollCallIndex,
    const RollCallParameters &initialParams,
    const VoteMatrix &votes,
    const Eigen::VectorXd &weights,
    const NormalCDF &normalCDF,
    const RollCallOptimizerConfig &config = RollCallOptimizerConfig());

#endif // ROLLCALL_OPTIMIZER_HPP

#ifndef GRID_OPTIMIZER_HPP
#define GRID_OPTIMIZER_HPP

/**
 * @brief Optimizadores por busqueda grid para parametros de DW-NOMINATE.
 *
 * Optimizadores de Parametros
 * - 2.1 SIGMAS() -> optimizeBeta()
 * - 2.2 WINT() -> optimizeWeights() [pendiente]
 *
 * Traduce las subrutinas de optimizacion unidimensional del codigo Fortran,
 * manteniendo el algoritmo de busqueda grid con backtracking adaptativo.
 */

#include "likelihood.hpp"
#include "normal_cdf.hpp"
#include <Eigen/Dense>
#include <vector>
#include <functional>

/**
 * @brief Resultado de la optimizacion de beta.
 */
struct BetaOptimizationResult
{
    double beta;          // Valor optimo de beta encontrado
    double logLikelihood; // Log-likelihood con beta optimo
    int iterations;       // Numero total de evaluaciones de LL
    double initialBeta;   // Valor inicial de beta
    double initialLL;     // Log-likelihood inicial
    bool converged;       // true si se encontro un optimo local
    int direction;        // 1 = ascendente, -1 = descendente, 0 = sin movimiento
};

/**
 * @brief Parametros de configuracion para optimizeBeta().
 *
 * Valores por defecto corresponden a los hardcoded en Fortran:
 * - NINC = 15 (maximo iteraciones por direccion)
 * - XINC = 0.1 (paso inicial)
 */
struct BetaOptimizerConfig
{
    int maxIterations;  // NINC: maximo iteraciones por direccion
    double initialStep; // XINC: tamano inicial del paso
    double minStep;     // Paso minimo antes de parar (derivado de XINC/2^15)
    bool verbose;       // Imprimir progreso (equivalente a WRITE comentados)

    /**
     * @brief Constructor con valores por defecto del Fortran.
     */
    BetaOptimizerConfig()
        : maxIterations(15),
          initialStep(0.1),
          minStep(0.1 / 32768.0), // 0.1 / 2^15 ~ 3e-6
          verbose(false)
    {
    }
};

/**
 * @brief Contexto de datos necesario para evaluar log-likelihood.
 *
 * Encapsula todos los datos que SIGMAS() accede via modulos globales
 * (xxcom_mod, mine_mod) en el codigo Fortran.
 */
struct LikelihoodContext
{
    const Eigen::MatrixXd &legislatorCoords;               // XDATA
    const std::vector<RollCallParameters> &rollCallParams; // ZMID, DYN
    const VoteMatrix &votes;                               // RCVOTE1, RCVOTE9
    Eigen::VectorXd &weights;                              // WEIGHT (mutable, contiene beta)
    const NormalCDF &normalCDF;                            // ZDF
    const std::vector<bool> &validRollCalls;               // RCBAD

    LikelihoodContext(
        const Eigen::MatrixXd &coords,
        const std::vector<RollCallParameters> &rcParams,
        const VoteMatrix &voteMatrix,
        Eigen::VectorXd &wts,
        const NormalCDF &cdf,
        const std::vector<bool> &valid)
        : legislatorCoords(coords),
          rollCallParams(rcParams),
          votes(voteMatrix),
          weights(wts),
          normalCDF(cdf),
          validRollCalls(valid)
    {
    }
};

/**
 * @brief Optimiza el parametro beta (sigma-squared inverso) via busqueda grid.
 */
BetaOptimizationResult optimizeBeta(
    LikelihoodContext &context,
    const BetaOptimizerConfig &config = BetaOptimizerConfig());

/**
 * @brief Version simplificada que retorna solo el log-likelihood optimizado.
 *
 * Equivalente funcional a la llamada Fortran:
 *   CALL SIGMAS(XPLOG, NFIRST, NLAST)
 *
 * @param context Contexto con datos del modelo
 * @return Log-likelihood con beta optimizado
 */
double optimizeBetaSimple(LikelihoodContext &context);

#endif 

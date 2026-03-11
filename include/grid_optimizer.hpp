#ifndef GRID_OPTIMIZER_HPP
#define GRID_OPTIMIZER_HPP

/**
 * Optimizadores por busqueda grid para parametros de DW-NOMINATE
 *
 * Optimizadores de Parametros
 * - SIGMAS() -> optimizeBeta()
 * - WINT() -> optimizeWeights()
 */

#include "likelihood.hpp"
#include "normal_cdf.hpp"
#include <Eigen/Dense>
#include <vector>
#include <functional>

/**
 * Resultado de la optimizacion de un parametro escalar
 * Estructura generica usada por optimizeBeta() y optimizeWeights()
 */
struct ParameterOptimizationResult
{
    double value;         // Valor optimo del parametro encontrado
    double logLikelihood; // Log-likelihood con parametro optimo
    int iterations;       // Numero total de evaluaciones de LL
    double initialValue;  // Valor inicial del parametro
    double initialLL;     // Log-likelihood inicial
    bool converged;       // true si se encontro un optimo local
    int direction;        // 1 = ascendente, -1 = descendente, 0 = sin movimiento
};

// Alias para compatibilidad con codigo existente
using BetaOptimizationResult = ParameterOptimizationResult;
using WeightOptimizationResult = ParameterOptimizationResult;

/**
 * Parametros de configuracion para optimizadores de busqueda grid.
 * Valores por defecto corresponden a SIGMAS:
 * - NINC = 15 (maximo iteraciones por direccion)
 * - XINC = 0.1 (paso inicial)
 *
 * Para WINT, usar initialStep = 0.01
 */
struct GridOptimizerConfig
{
    int maxIterations;  // NINC: maximo iteraciones por direccion
    double initialStep; // XINC: tamano inicial del paso
    double minStep;     // Paso minimo antes de parar
    bool verbose;       // Imprimir progreso

    // Constructor con valores por defecto (SIGMAS).
    GridOptimizerConfig()
        : maxIterations(15),
          initialStep(0.1),
          minStep(0.1 / 32768.0), // 0.1 / 2^15 ~ 3e-6
          verbose(false)
    {
    }

    /**
     * Constructor con paso inicial personalizado.
     * @param step Paso inicial (0.1 para SIGMAS, 0.01 para WINT)
     */
    explicit GridOptimizerConfig(double step)
        : maxIterations(15),
          initialStep(step),
          minStep(step / 32768.0),
          verbose(false)
    {
    }
};

// Alias para compatibilidad
using BetaOptimizerConfig = GridOptimizerConfig;
using WeightOptimizerConfig = GridOptimizerConfig;

/**
 * Contexto de datos necesario para evaluar log-likelihood
 * Encapsula todos los datos que SIGMAS()/WINT() acceden via modulos globales
 * (xxcom_mod, mine_mod) en el codigo Fortran
 */
struct LikelihoodContext
{
    const Eigen::MatrixXd &legislatorCoords;               // XDATA
    const std::vector<RollCallParameters> &rollCallParams; // ZMID, DYN
    const VoteMatrix &votes;                               // RCVOTE1, RCVOTE9
    Eigen::VectorXd &weights;                              // WEIGHT (mutable)
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
 * Optimiza un parametro escalar via busqueda grid con backtracking.
 * Funcion generica que implementa el algoritmo comun a SIGMAS() y WINT().
 *
 * Algoritmo:
 * 1. Evalua LL en param_actual, param+step, param-step
 * 2. Determina direccion de mejora (arriba o abajo)
 * 3. Itera en esa direccion:
 *    - Si LL mejora: continua avanzando
 *    - Si LL empeora: retrocede y reduce paso a la mitad
 * 4. Repite hasta maxIterations pasos
 *
 * @param context Contexto con datos del modelo (weights es modificado)
 * @param paramIndex Indice del parametro a optimizar en weights[]
 * @param config Parametros de configuracion
 * @return Resultado con valor optimo y estadisticas
 */
ParameterOptimizationResult optimizeParameter(
    LikelihoodContext &context,
    int paramIndex,
    const GridOptimizerConfig &config);

/**
 * Optimiza el parametro beta (sigma-squared inverso).
 * @param context Contexto con datos del modelo
 * @param config Configuracion (por defecto: step=0.1, maxIter=15)
 * @return Resultado con beta optimo
 *
 */
BetaOptimizationResult optimizeBeta(
    LikelihoodContext &context,
    const BetaOptimizerConfig &config = BetaOptimizerConfig());

/**
 * Optimiza el peso de la segunda dimension (W2)
 * IMPORTANTE: Solo debe llamarse cuando NS >= 2 (modelos multidimensionales)
 * En modelos unidimensionales (NS=1), WEIGHT(2) no existe
 * @param context Contexto con datos del modelo
 * @param config Configuracion (por defecto: step=0.01, maxIter=15)
 * @return Resultado con W2 optimo
 *
 */
WeightOptimizationResult optimizeWeight2(
    LikelihoodContext &context,
    const WeightOptimizerConfig &config = WeightOptimizerConfig(0.01));

// Configuracion por defecto para optimizacion de beta (SIGMAS)
inline GridOptimizerConfig sigmasConfig()
{
    return GridOptimizerConfig(0.1);
}

// Configuracion por defecto para optimizacion de pesos (WINT)
inline GridOptimizerConfig wintConfig()
{
    return GridOptimizerConfig(0.01);
}

#endif // GRID_OPTIMIZER_HPP

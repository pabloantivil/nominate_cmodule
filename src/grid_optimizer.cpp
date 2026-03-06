/**
 * Implementacion de optimizadores por busqueda grid para DW-NOMINATE.
 *
 * Optimizadores de Parametros
 * - SIGMAS() -> optimizeBeta()
 * - WINT() -> optimizeWeight2()
 *
 * Diferencias:
 * - SIGMAS: optimiza WEIGHT(NS+1) con paso 0.1
 * - WINT: optimiza WEIGHT(2) con paso 0.01
 *
 * OPTIMIZACIONES IMPLEMENTADAS:
 * - Buffer de trabajo reutilizable para evaluaciones de likelihood
 * - Evita allocations dinamicas en hot loops
 */

#include "grid_optimizer.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <stdexcept>

// Buffer global de trabajo para evaluaciones (CORRECCION B)
// Thread-local para seguridad en caso de uso con OpenMP
static thread_local LikelihoodWorkBuffer g_likelihoodBuffer;

/**
 * Evalua log-likelihood usando version OPTIMIZADA.
 * OPTIMIZADO: Usa buffer reutilizable para evitar allocations.
 */
static double evaluateLogLikelihood(const LikelihoodContext &ctx)
{
    const int numDimensions = static_cast<int>(ctx.weights.size()) - 1;

    if (ctx.legislatorCoords.cols() != numDimensions)
    {
        throw std::invalid_argument(
            "Dimensiones inconsistentes entre weights y legislatorCoords");
    }

    // OPTIMIZADO: Usar version con buffer pre-allocated
    LikelihoodResult result = computeLogLikelihoodOptimized(
        ctx.legislatorCoords,
        ctx.rollCallParams,
        ctx.votes,
        ctx.weights,
        ctx.normalCDF,
        ctx.validRollCalls,
        g_likelihoodBuffer);

    return result.logLikelihood;
}

// ============================================================================
// Implementacion de optimizeParameter()
// ============================================================================

ParameterOptimizationResult optimizeParameter(
    LikelihoodContext &context,
    int paramIndex,
    const GridOptimizerConfig &config)
{
    /**
     * Algoritmo comun a SIGMAS() y WINT() del Fortran.
     *
     * Estructura:
     * 1. Evalua LL en param_actual -> SAVECURR
     * 2. Evalua LL en param+XINC -> SAVEUP
     * 3. Evalua LL en param-XINC -> SAVEDWN
     * 4. Si SAVEUP > SAVECURR: busqueda ascendente
     * 5. Si SAVEDWN > SAVECURR: busqueda descendente
     * 6. En cada iteracion:
     *    - Si LL mejora: acepta y continua
     *    - Si LL empeora: rechaza y XINC = XINC/2
     */

    // Validar indice
    if (paramIndex < 0 || paramIndex >= static_cast<int>(context.weights.size()))
    {
        throw std::out_of_range("paramIndex fuera de rango en optimizeParameter");
    }

    ParameterOptimizationResult result;
    result.initialValue = context.weights(paramIndex);
    result.iterations = 0;
    result.converged = false;
    result.direction = 0;

    double xinc = config.initialStep;

    // Evaluar punto actual
    double saveCurr = evaluateLogLikelihood(context);
    result.initialLL = saveCurr;
    result.iterations++;

    if (config.verbose)
    {
        std::cout << "[optimizeParameter] Inicial: param[" << paramIndex << "]="
                  << result.initialValue
                  << ", LL=" << std::fixed << std::setprecision(4) << saveCurr
                  << std::endl;
    }

    // Sondeo hacia arriba (+XINC)
    context.weights(paramIndex) += xinc;
    double saveUp = evaluateLogLikelihood(context);
    result.iterations++;
    context.weights(paramIndex) -= xinc; // Restaurar

    // Sondeo hacia abajo (-XINC)
    context.weights(paramIndex) -= xinc;
    double saveDwn = evaluateLogLikelihood(context);
    result.iterations++;
    context.weights(paramIndex) += xinc; // Restaurar

    if (config.verbose)
    {
        std::cout << "[optimizeParameter] Sondeo: LL_up=" << saveUp
                  << ", LL_dwn=" << saveDwn << std::endl;
    }

    // Caso A: Busqueda ascendente (SAVEUP > SAVECURR)
    if (saveUp > saveCurr)
    {
        result.direction = 1;

        context.weights(paramIndex) += xinc;
        saveCurr = evaluateLogLikelihood(context);
        result.iterations++;

        for (int iter = 0; iter < config.maxIterations; ++iter)
        {
            context.weights(paramIndex) += xinc;
            saveUp = evaluateLogLikelihood(context);
            result.iterations++;

            if (config.verbose)
            {
                std::cout << "[optimizeParameter] Iter " << (iter + 1)
                          << " UP: param=" << context.weights(paramIndex)
                          << ", LL=" << saveUp
                          << ", step=" << xinc << std::endl;
            }

            if (saveUp < saveCurr)
            {
                context.weights(paramIndex) -= xinc;
                xinc = xinc / 2.0;

                if (config.verbose)
                {
                    std::cout << "[optimizeParameter]   Empeoro. Retrocede, step="
                              << xinc << std::endl;
                }
            }

            if (saveUp > saveCurr)
            {
                saveCurr = saveUp;

                if (config.verbose)
                {
                    std::cout << "[optimizeParameter]   Mejoro. LL=" << saveCurr << std::endl;
                }
            }

            if (xinc < config.minStep)
            {
                result.converged = true;
                break;
            }
        }
    }

    // Caso B: Busqueda descendente (SAVEDWN > SAVECURR)
    if (saveDwn > saveCurr)
    {
        result.direction = -1;

        context.weights(paramIndex) -= xinc;
        saveCurr = evaluateLogLikelihood(context);
        result.iterations++;

        for (int iter = 0; iter < config.maxIterations; ++iter)
        {
            context.weights(paramIndex) -= xinc;
            saveDwn = evaluateLogLikelihood(context);
            result.iterations++;

            if (config.verbose)
            {
                std::cout << "[optimizeParameter] Iter " << (iter + 1)
                          << " DWN: param=" << context.weights(paramIndex)
                          << ", LL=" << saveDwn
                          << ", step=" << xinc << std::endl;
            }

            if (saveDwn < saveCurr)
            {
                context.weights(paramIndex) += xinc;
                xinc = xinc / 2.0;

                if (config.verbose)
                {
                    std::cout << "[optimizeParameter]   Empeoro. Retrocede, step="
                              << xinc << std::endl;
                }
            }

            if (saveDwn > saveCurr)
            {
                saveCurr = saveDwn;

                if (config.verbose)
                {
                    std::cout << "[optimizeParameter]   Mejoro. LL=" << saveCurr << std::endl;
                }
            }

            if (xinc < config.minStep)
            {
                result.converged = true;
                break;
            }
        }
    }

    // Caso C: Sin movimiento
    if (result.direction == 0)
    {
        result.converged = true;

        if (config.verbose)
        {
            std::cout << "[optimizeParameter] Sin movimiento. Ya en optimo local."
                      << std::endl;
        }
    }

    // Resultado final
    result.value = context.weights(paramIndex);
    result.logLikelihood = saveCurr;

    if (config.verbose)
    {
        std::cout << "[optimizeParameter] Final: param=" << result.value
                  << ", LL=" << result.logLikelihood
                  << ", iters=" << result.iterations
                  << ", dir=" << result.direction
                  << std::endl;
    }

    return result;
}

// Implementacion de optimizeBeta() - SIGMAS
BetaOptimizationResult optimizeBeta(
    LikelihoodContext &context,
    const BetaOptimizerConfig &config)
{
    // Indice de beta: ultima posicion del vector weights
    // Fortran: WEIGHT(NS+1) donde NS = numero de dimensiones
    const int betaIndex = static_cast<int>(context.weights.size()) - 1;

    // Usar funcion generica
    ParameterOptimizationResult genResult = optimizeParameter(context, betaIndex, config);

    // Adaptar resultado al tipo esperado
    BetaOptimizationResult result;
    result.value = genResult.value;
    result.logLikelihood = genResult.logLikelihood;
    result.iterations = genResult.iterations;
    result.initialValue = genResult.initialValue;
    result.initialLL = genResult.initialLL;
    result.converged = genResult.converged;
    result.direction = genResult.direction;

    return result;
}

// Implementacion de optimizeBetaSimple()
double optimizeBetaSimple(LikelihoodContext &context)
{
    BetaOptimizerConfig config = sigmasConfig();
    config.verbose = false;

    BetaOptimizationResult result = optimizeBeta(context, config);

    return result.logLikelihood;
}

// Implementacion de optimizeWeight2() - WINT
WeightOptimizationResult optimizeWeight2(
    LikelihoodContext &context,
    const WeightOptimizerConfig &config)
{
    // Verificar que el modelo tiene al menos 2 dimensiones
    const int numDimensions = static_cast<int>(context.weights.size()) - 1;

    if (numDimensions < 2)
    {
        throw std::invalid_argument(
            "optimizeWeight2 requiere NS >= 2 (modelo multidimensional). "
            "Actual: NS = " +
            std::to_string(numDimensions));
    }

    // Indice de W2: posicion 1 del vector weights (0-based)
    const int w2Index = 1;

    // Usar funcion generica
    ParameterOptimizationResult genResult = optimizeParameter(context, w2Index, config);

    // Adaptar resultado
    WeightOptimizationResult result;
    result.value = genResult.value;
    result.logLikelihood = genResult.logLikelihood;
    result.iterations = genResult.iterations;
    result.initialValue = genResult.initialValue;
    result.initialLL = genResult.initialLL;
    result.converged = genResult.converged;
    result.direction = genResult.direction;

    return result;
}

// Implementacion de optimizeWeight2Simple() - WINT
double optimizeWeight2Simple(LikelihoodContext &context)
{
    WeightOptimizerConfig config = wintConfig();
    config.verbose = false;

    WeightOptimizationResult result = optimizeWeight2(context, config);

    return result.logLikelihood;
}

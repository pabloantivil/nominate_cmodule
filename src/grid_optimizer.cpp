/**
 * @brief Implementacion de optimizadores por busqueda grid para DW-NOMINATE.
 */

#include "grid_optimizer.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>

// ============================================================================
// Funcion auxiliar: evalua log-likelihood con el estado actual
// ============================================================================

/**
 * @brief Evalua log-likelihood usando computeLogLikelihood().
 *
 * Equivalente a: CALL PLOG(XPLOG, NFIRST, NLAST)
 *
 * @param ctx Contexto con datos del modelo
 * @return Log-likelihood total (XPLOG)
 */
static double evaluateLogLikelihood(const LikelihoodContext &ctx)
{
    // Numero de dimensiones = weights.size() - 1 (ultimo es beta)
    const int numDimensions = static_cast<int>(ctx.weights.size()) - 1;

    // Verifica que las coordenadas tengan el numero correcto de dimensiones
    if (ctx.legislatorCoords.cols() != numDimensions)
    {
        throw std::invalid_argument(
            "Dimensiones inconsistentes entre weights y legislatorCoords");
    }

    LikelihoodResult result = computeLogLikelihood(
        ctx.legislatorCoords,
        ctx.rollCallParams,
        ctx.votes,
        ctx.weights,
        ctx.normalCDF,
        ctx.validRollCalls);

    return result.logLikelihood;
}

// ============================================================================
// Implementacion de optimizeBeta()
// ============================================================================

BetaOptimizationResult optimizeBeta(
    LikelihoodContext &context,
    const BetaOptimizerConfig &config)
{
    /**
     * SUBROUTINE SIGMAS
     *
     * Estructura del algoritmo Fortran:
     * 1. NINC=15, XINC=0.1  (parametros hardcoded)
     * 2. Evalua LL en beta_actual -> SAVECURR
     * 3. Evalua LL en beta+XINC -> SAVEUP
     * 4. Evalua LL en beta-XINC -> SAVEDWN
     * 5. Si SAVEUP > SAVECURR: busqueda ascendente
     * 6. Si SAVEDWN > SAVECURR: busqueda descendente
     * 7. En cada iteracion:
     *    - Si LL mejora: acepta y continua
     *    - Si LL empeora: rechaza y XINC = XINC/2
     */

    BetaOptimizationResult result;

    // Indice de beta en el vector de pesos: ultima posicion
    // Fortran: WEIGHT(NS+1) donde NS = numero de dimensiones
    const int betaIndex = static_cast<int>(context.weights.size()) - 1;

    // Guardar estado inicial
    result.initialBeta = context.weights(betaIndex);
    result.iterations = 0;

    // Paso de busqueda (mutable durante la optimizacion)
    // Fortran: XINC = 0.1
    double xinc = config.initialStep;

    // ========================================================================
    // Bloque 1: Evaluar punto actual
    //   CALL PLOG(XPLOG, NFIRST, NLAST)
    //   SAVECURR = XPLOG
    // ========================================================================

    double saveCurr = evaluateLogLikelihood(context);
    result.initialLL = saveCurr;
    result.iterations++;

    if (config.verbose)
    {
        std::cout << "[optimizeBeta] Inicial: beta=" << result.initialBeta
                  << ", LL=" << std::fixed << std::setprecision(3) << saveCurr
                  << std::endl;
    }

    // ========================================================================
    // Bloque 2: Sondeo hacia arriba (+XINC)
    //   WEIGHT(NS+1) = WEIGHT(NS+1) + XINC
    //   CALL PLOG(XPLOG, NFIRST, NLAST)
    //   SAVEUP = XPLOG
    //   WEIGHT(NS+1) = WEIGHT(NS+1) - XINC
    // ========================================================================

    context.weights(betaIndex) += xinc;
    double saveUp = evaluateLogLikelihood(context);
    result.iterations++;
    context.weights(betaIndex) -= xinc; // Restaurar

    // ========================================================================
    // Bloque 3: Sondeo hacia abajo (-XINC)
    //   WEIGHT(NS+1) = WEIGHT(NS+1) - XINC
    //   CALL PLOG(XPLOG, NFIRST, NLAST)
    //   SAVEDWN = XPLOG
    //   WEIGHT(NS+1) = WEIGHT(NS+1) + XINC
    // ========================================================================

    context.weights(betaIndex) -= xinc;
    double saveDwn = evaluateLogLikelihood(context);
    result.iterations++;
    context.weights(betaIndex) += xinc; // Restaurar

    if (config.verbose)
    {
        std::cout << "[optimizeBeta] Sondeo: LL_up=" << saveUp
                  << ", LL_dwn=" << saveDwn << std::endl;
    }

    // ========================================================================
    // Bloque 4: Determinar direccion de busqueda
    // ========================================================================

    result.converged = false;
    result.direction = 0;

    // Caso A: Busqueda ascendente (SAVEUP > SAVECURR)

    if (saveUp > saveCurr)
    {
        result.direction = 1;
        context.weights(betaIndex) += xinc;
        saveCurr = evaluateLogLikelihood(context);
        result.iterations++;

        // DO 1 I=1,NINC
        for (int iter = 0; iter < config.maxIterations; ++iter)
        {
            // WEIGHT(NS+1) = WEIGHT(NS+1) + XINC
            context.weights(betaIndex) += xinc;

            // CALL PLOG(XPLOG, NFIRST, NLAST)
            // SAVEUP = XPLOG
            saveUp = evaluateLogLikelihood(context);
            result.iterations++;

            if (config.verbose)
            {
                std::cout << "[optimizeBeta] Iter " << (iter + 1)
                          << " UP: beta=" << context.weights(betaIndex)
                          << ", LL=" << saveUp
                          << ", step=" << xinc << std::endl;
            }

            // IF(SAVEUP.LT.SAVECURR) THEN
            //   WEIGHT(NS+1) = WEIGHT(NS+1) - XINC
            //   XINC = XINC / 2.0
            // ENDIF
            if (saveUp < saveCurr)
            {
                // Empeoro: retroceder y reducir paso
                context.weights(betaIndex) -= xinc;
                xinc = xinc / 2.0;

                if (config.verbose)
                {
                    std::cout << "[optimizeBeta]   Empeoro. Retrocede, nuevo step="
                              << xinc << std::endl;
                }
            }

            // IF(SAVEUP.GT.SAVECURR) THEN
            //   SAVECURR = SAVEUP
            // ENDIF
            if (saveUp > saveCurr)
            {
                // Mejoro: actualizar referencia
                saveCurr = saveUp;

                if (config.verbose)
                {
                    std::cout << "[optimizeBeta]   Mejoro. Nueva ref LL="
                              << saveCurr << std::endl;
                }
            }

            // Early exit si el paso es muy pequeno
            if (xinc < config.minStep)
            {
                result.converged = true;
                break;
            }
        }
    }

    // ========================================================================
    // Caso B: Busqueda descendente (SAVEDWN > SAVECURR)
    // ========================================================================

    if (saveDwn > saveCurr)
    {
        result.direction = -1;

        //   WEIGHT(NS+1) = WEIGHT(NS+1) - XINC
        //   CALL PLOG(XPLOG, NFIRST, NLAST)
        //   SAVECURR = XPLOG
        context.weights(betaIndex) -= xinc;
        saveCurr = evaluateLogLikelihood(context);
        result.iterations++;

        // DO 2 I=1,NINC
        for (int iter = 0; iter < config.maxIterations; ++iter)
        {
            // WEIGHT(NS+1) = WEIGHT(NS+1) - XINC
            context.weights(betaIndex) -= xinc;

            // CALL PLOG(XPLOG, NFIRST, NLAST)
            // SAVEDWN = XPLOG
            saveDwn = evaluateLogLikelihood(context);
            result.iterations++;

            if (config.verbose)
            {
                std::cout << "[optimizeBeta] Iter " << (iter + 1)
                          << " DWN: beta=" << context.weights(betaIndex)
                          << ", LL=" << saveDwn
                          << ", step=" << xinc << std::endl;
            }

            // IF(SAVEDWN.LT.SAVECURR) THEN
            //   WEIGHT(NS+1) = WEIGHT(NS+1) + XINC
            //   XINC = XINC / 2.0
            // ENDIF
            if (saveDwn < saveCurr)
            {
                // Empeoro: retroceder y reducir paso
                context.weights(betaIndex) += xinc;
                xinc = xinc / 2.0;

                if (config.verbose)
                {
                    std::cout << "[optimizeBeta]   Empeoro. Retrocede, nuevo step="
                              << xinc << std::endl;
                }
            }

            // IF(SAVEDWN.GT.SAVECURR) THEN
            //   SAVECURR = SAVEDWN
            // ENDIF
            if (saveDwn > saveCurr)
            {
                // Mejoro: actualizar referencia
                saveCurr = saveDwn;

                if (config.verbose)
                {
                    std::cout << "[optimizeBeta]   Mejoro. Nueva ref LL="
                              << saveCurr << std::endl;
                }
            }

            // Early exit si el paso es muy pequeno
            if (xinc < config.minStep)
            {
                result.converged = true;
                break;
            }
        }
    }

    // ========================================================================
    // Caso C: Sin movimiento (ningun sondeo mejoro)
    // ========================================================================

    if (result.direction == 0)
    {
        // Beta inicial ya era optimo local
        result.converged = true;

        if (config.verbose)
        {
            std::cout << "[optimizeBeta] Sin movimiento. Beta inicial es optimo local."
                      << std::endl;
        }
    }

    // Resultado final

    result.beta = context.weights(betaIndex);
    result.logLikelihood = saveCurr;

    if (config.verbose)
    {
        std::cout << "[optimizeBeta] Final: beta=" << result.beta
                  << ", LL=" << result.logLikelihood
                  << ", iters=" << result.iterations
                  << ", dir=" << result.direction
                  << std::endl;
    }

    return result;
}

// ============================================================================
// Implementacion de optimizeBetaSimple()
// ============================================================================

double optimizeBetaSimple(LikelihoodContext &context)
{
    BetaOptimizerConfig config;
    config.verbose = false;

    BetaOptimizationResult result = optimizeBeta(context, config);

    return result.logLikelihood;
}

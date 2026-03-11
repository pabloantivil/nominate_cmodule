/**
 * Implementacion del optimizador de parametros de roll calls (RCINT2).
 *
 * OPTIMIZACIONES IMPLEMENTADAS:
 * - Buffer de trabajo reutilizable para evaluaciones de derivadas
 * - Estructuras de tamano fijo para evitar allocations en hot loops
 * - Uso de versiones optimizadas de computeRollCallDerivatives
 */

#include "rollcall_optimizer.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <array>

// Buffer global de trabajo
// Thread-local para seguridad en caso de uso con OpenMP
static thread_local RollCallDerivativesWorkBuffer g_rcDerivBuffer;

/**
 * Estado de un punto en la busqueda lineal - VERSION OPTIMIZADA.
 * Usa arrays de tamano fijo en lugar de VectorXd.
 */
struct SearchPointOptimized
{
    std::array<double, MAX_DIMENSIONS> midpoint;
    std::array<double, MAX_DIMENSIONS> spread;
    double gmp;
    double logLikelihood;
    int index;

    SearchPointOptimized() : gmp(0.0), logLikelihood(0.0), index(0)
    {
        midpoint.fill(0.0);
        spread.fill(0.0);
    }
};

// Estructuras originales mantenidas para compatibilidad
struct SearchPoint
{
    Eigen::VectorXd midpoint;
    Eigen::VectorXd spread;
    double gmp;
    double logLikelihood;
    int index;

    SearchPoint() : gmp(0.0), logLikelihood(0.0), index(0) {}

    explicit SearchPoint(int numDim)
        : midpoint(Eigen::VectorXd::Zero(numDim)),
          spread(Eigen::VectorXd::Zero(numDim)),
          gmp(0.0),
          logLikelihood(0.0),
          index(0)
    {
    }
};

// Estado guardado para backtracking - VERSION OPTIMIZADA.
struct SavedStateOptimized
{
    std::array<double, MAX_DIMENSIONS> midpoint;
    std::array<double, MAX_DIMENSIONS> spread;
    int numDim;

    SavedStateOptimized() : numDim(0)
    {
        midpoint.fill(0.0);
        spread.fill(0.0);
    }

    void save(const Eigen::VectorXd &mid, const Eigen::VectorXd &spr)
    {
        numDim = static_cast<int>(mid.size());
        for (int k = 0; k < numDim && k < MAX_DIMENSIONS; ++k)
        {
            midpoint[k] = mid(k);
            spread[k] = spr(k);
        }
    }
};

// Estructura original para compatibilidad
struct SavedState
{
    Eigen::VectorXd midpoint;
    Eigen::VectorXd spread;

    SavedState() = default;

    explicit SavedState(int numDim)
        : midpoint(Eigen::VectorXd::Zero(numDim)),
          spread(Eigen::VectorXd::Zero(numDim))
    {
    }

    void save(const Eigen::VectorXd &mid, const Eigen::VectorXd &spr)
    {
        midpoint = mid;
        spread = spr;
    }
};

// Funciones auxiliares

// Protege spread contra valores muy pequenios.
static void protectSpread(
    Eigen::VectorXd &spread,
    double minSpread,
    double defaultSpread)
{
    for (int k = 0; k < spread.size(); ++k)
    {
        if (std::abs(spread(k)) <= minSpread)
        {
            spread(k) = defaultSpread;
        }
    }
}

/**
 * Verifica si spread tiene valores protegidos.
 * Retorna true si algun spread fue reseteado.
 */
static bool checkAndProtectSpread(
    Eigen::VectorXd &spread,
    double minSpread,
    double defaultSpread)
{
    bool wasProtected = false;
    for (int k = 0; k < spread.size(); ++k)
    {
        if (std::abs(spread(k)) <= minSpread)
        {
            spread(k) = defaultSpread;
            wasProtected = true;
        }
    }
    return wasProtected;
}

/**
 * Calcula el tamanio de paso normalizado.
 * @param gradientNormSquared Norma al cuadrado del gradiente (SUMB)
 * @param stepUnit Unidad base (0.01)
 * @return Tamano de paso
 */
static double computeStepSize(double gradientNormSquared, double stepUnit)
{
    if (gradientNormSquared < 1e-20)
    {
        return stepUnit; // Evitar division por cero
    }
    return stepUnit / std::sqrt(gradientNormSquared);
}

// Verifica si todas las derivadas son esencialmente cero.
static bool areDerivativesZero(
    const Eigen::VectorXd &derivatives,
    double tolerance)
{
    for (int k = 0; k < derivatives.size(); ++k)
    {
        if (std::abs(derivatives(k)) > tolerance)
        {
            return false;
        }
    }
    return true;
}

/**
 * Proyecta midpoint a la superficie de la hiperesfera unitaria.
 * @param midpoint Vector a proyectar (modificado in-place)
 * @return true si se aplico proyeccion
 */
static bool projectToUnitSphere(Eigen::VectorXd &midpoint)
{
    double normSquared = midpoint.squaredNorm();
    if (normSquared > 1.0)
    {
        midpoint /= std::sqrt(normSquared);
        return true;
    }
    return false;
}

/**
 * Encuentra el indice del punto con mayor GMP.
 * Usa el patron de RSORT pero simplificado para encontrar el maximo.
 */
static int findBestPoint(const std::vector<SearchPoint> &points, int numValid)
{
    if (numValid <= 0)
        return 0;

    int bestIdx = 0;
    double bestGMP = points[0].gmp;

    for (int i = 1; i < numValid; ++i)
    {
        if (points[i].gmp > bestGMP)
        {
            bestGMP = points[i].gmp;
            bestIdx = i;
        }
    }
    return bestIdx;
}

// Fase A: Optimizacion de Spread (DYN)

/**
 *  Ejecuta una iteracion de optimizacion de spread.
 *  VERSION OPTIMIZADA: Usa buffer global y funcion optimizada
 * @return GMP alcanzado en esta iteracion
 */
static double optimizeSpreadIteration(
    const Eigen::MatrixXd &legislatorCoords,
    int rollCallIndex,
    Eigen::VectorXd &midpoint,
    Eigen::VectorXd &spread,
    const VoteMatrix &votes,
    const Eigen::VectorXd &weights,
    const NormalCDF &normalCDF,
    const RollCallOptimizerConfig &config,
    bool &exitEarly)
{
    const int numDim = static_cast<int>(midpoint.size());
    exitEarly = false;

    // Guardar estado actual usando arrays fijos
    SavedStateOptimized saved;
    saved.save(midpoint, spread);

    // Proteger spread contra valores extremos
    protectSpread(spread, config.minSpread, config.defaultSpread);
    for (int k = 0; k < numDim && k < MAX_DIMENSIONS; ++k)
    {
        saved.spread[k] = spread(k);
    }

    // Calcular derivadas con PROLLC2 OPTIMIZADO
    auto derivResult = computeRollCallDerivativesOptimized(
        legislatorCoords, rollCallIndex, midpoint, spread,
        votes, weights, normalCDF, g_rcDerivBuffer);

    int numVotes = derivResult.totalVotes;
    if (numVotes == 0)
    {
        exitEarly = true;
        return 0.0;
    }

    double currentGMP = derivResult.geometricMeanProb;

    // Normalizar gradiente usando arrays fijos
    std::array<double, MAX_DIMENSIONS> gradSpread;
    double gradNormSq = 0.0;
    for (int k = 0; k < numDim && k < MAX_DIMENSIONS; ++k)
    {
        gradSpread[k] = derivResult.spreadDerivatives(k) / static_cast<double>(numVotes);
        gradNormSq += gradSpread[k] * gradSpread[k];
    }

    // Verificar derivadas nulas
    bool allZero = true;
    for (int k = 0; k < numDim && k < MAX_DIMENSIONS; ++k)
    {
        if (std::abs(gradSpread[k]) > config.derivativeTol)
        {
            allZero = false;
            break;
        }
    }
    if (allZero)
    {
        exitEarly = true;
        return currentGMP;
    }

    // Calcular tamanio de paso
    double stepSize = computeStepSize(gradNormSq, config.stepUnit);

    // Busqueda lineal con arrays fijos
    // Preallocamos solo numSearchPoints estructuras livianas
    std::array<SearchPointOptimized, 15> searchPointsFixed; // Max 15 puntos (NINC tipico)
    const int numPts = std::min(config.numSearchPoints, 15);
    double stepAccum = 0.0;

    double bestGMP = -1e30;
    int bestIdx = 0;

    for (int kk = 0; kk < numPts; ++kk)
    {
        // Actualizar spread en direccion del gradiente negativo
        for (int k = 0; k < numDim && k < MAX_DIMENSIONS; ++k)
        {
            spread(k) = saved.spread[k] - stepAccum * gradSpread[k];
        }

        // Evaluar en este punto
        auto evalResult = computeRollCallDerivativesOptimized(
            legislatorCoords, rollCallIndex, midpoint, spread,
            votes, weights, normalCDF, g_rcDerivBuffer);

        // Guardar y comparar inline (evita almacenar todo)
        double gmp = evalResult.geometricMeanProb;
        if (gmp > bestGMP)
        {
            bestGMP = gmp;
            bestIdx = kk;
            for (int k = 0; k < numDim && k < MAX_DIMENSIONS; ++k)
            {
                searchPointsFixed[kk].midpoint[k] = midpoint(k);
                searchPointsFixed[kk].spread[k] = spread(k);
            }
            searchPointsFixed[kk].gmp = gmp;
        }

        stepAccum += stepSize;
    }

    // Actualizar parametros al mejor punto
    for (int k = 0; k < numDim && k < MAX_DIMENSIONS; ++k)
    {
        midpoint(k) = searchPointsFixed[bestIdx].midpoint[k];
        spread(k) = searchPointsFixed[bestIdx].spread[k];
    }

    // Proteger spread despues de actualizacion
    if (checkAndProtectSpread(spread, config.minSpread, config.defaultSpread))
    {
        exitEarly = true;
    }

    return bestGMP;
}

/**
 * Ejecuta la fase completa de optimizacion de spread.
 * VERSION OPTIMIZADA.
 */
static int optimizeSpreadPhase(
    const Eigen::MatrixXd &legislatorCoords,
    int rollCallIndex,
    Eigen::VectorXd &midpoint,
    Eigen::VectorXd &spread,
    const VoteMatrix &votes,
    const Eigen::VectorXd &weights,
    const NormalCDF &normalCDF,
    const RollCallOptimizerConfig &config,
    double &initialGMP)
{
    std::vector<double> gmpHistory(config.numInnerIterations, 0.0);
    int iterations = 0;

    for (int iter = 0; iter < config.numInnerIterations; ++iter)
    {
        bool exitEarly = false;
        double gmp = optimizeSpreadIteration(
            legislatorCoords, rollCallIndex, midpoint, spread,
            votes, weights, normalCDF, config, exitEarly);

        gmpHistory[iter] = gmp;
        iterations = iter + 1;

        // Guardar GMP inicial
        if (iter == 0)
        {
            initialGMP = gmp;
        }

        if (exitEarly)
        {
            break;
        }

        // Verificar convergencia despues de la 3ra iteracion
        if (iter >= 2)
        {
            double improvement = gmpHistory[iter] - gmpHistory[iter - 1];
            if (improvement <= config.convergenceTol)
            {
                break;
            }
        }
    }

    // OPTIMIZACION: Eliminada llamada redundante a computeRollCallDerivativesOptimized
    // (resultado no se usaba)

    return iterations;
}

// Fase B: Optimizacion de Midpoint (ZMID)

/**
 * Ejecuta una iteracion de optimizacion de midpoint.
 * VERSION OPTIMIZADA: Usa buffer global y arrays fijos.
 * Incluye restriccion de hiperesfera unitaria.
 */
static double optimizeMidpointIteration(
    const Eigen::MatrixXd &legislatorCoords,
    int rollCallIndex,
    Eigen::VectorXd &midpoint,
    Eigen::VectorXd &spread,
    const VoteMatrix &votes,
    const Eigen::VectorXd &weights,
    const NormalCDF &normalCDF,
    const RollCallOptimizerConfig &config,
    int &validPoints)
{
    const int numDim = static_cast<int>(midpoint.size());

    // Guardar estado actual con arrays fijos (CORRECCION A)
    SavedStateOptimized saved;
    saved.save(midpoint, spread);

    // Calcular derivadas con funcion optimizada (CORRECCION E)
    auto derivResult = computeRollCallDerivativesOptimized(
        legislatorCoords, rollCallIndex, midpoint, spread,
        votes, weights, normalCDF, g_rcDerivBuffer);

    int numVotes = derivResult.totalVotes;
    if (numVotes == 0)
    {
        validPoints = 0;
        return 0.0;
    }

    // Normalizar gradiente usando arrays fijos (CORRECCION A)
    std::array<double, MAX_DIMENSIONS> gradMidpoint;
    double gradNormSq = 0.0;
    for (int k = 0; k < numDim && k < MAX_DIMENSIONS; ++k)
    {
        gradMidpoint[k] = derivResult.midpointDerivatives(k) / static_cast<double>(numVotes);
        gradNormSq += gradMidpoint[k] * gradMidpoint[k];
    }

    // Calcular tamanio de paso
    double stepSize = computeStepSize(gradNormSq, config.stepUnit);

    // Busqueda lineal con arrays fijos y restriccion esferica
    std::array<SearchPointOptimized, 15> searchPointsFixed;
    const int numPts = std::min(config.numSearchPoints, 15);
    double stepAccum = 0.0;
    validPoints = 0;

    double bestGMP = -1e30;
    int bestIdx = 0;

    for (int kk = 0; kk < numPts; ++kk)
    {
        // Actualizar midpoint en direccion del gradiente negativo
        for (int k = 0; k < numDim && k < MAX_DIMENSIONS; ++k)
        {
            midpoint(k) = saved.midpoint[k] - stepAccum * gradMidpoint[k];
        }

        // Verificar restriccion de hiperesfera unitaria
        bool projected = projectToUnitSphere(midpoint);

        // Evaluar en este punto (CORRECCION E: funcion optimizada)
        auto evalResult = computeRollCallDerivativesOptimized(
            legislatorCoords, rollCallIndex, midpoint, spread,
            votes, weights, normalCDF, g_rcDerivBuffer);

        // Guardar y comparar inline
        double gmp = evalResult.geometricMeanProb;
        if (gmp > bestGMP)
        {
            bestGMP = gmp;
            bestIdx = kk;
            for (int k = 0; k < numDim && k < MAX_DIMENSIONS; ++k)
            {
                searchPointsFixed[kk].midpoint[k] = midpoint(k);
                searchPointsFixed[kk].spread[k] = spread(k);
            }
            searchPointsFixed[kk].gmp = gmp;
        }

        validPoints = kk + 1;

        // Si se proyecto, terminar busqueda (como en Fortran GO TO 2113)
        if (projected)
        {
            break;
        }

        stepAccum += stepSize;
    }

    // Actualizar parametros al mejor punto
    for (int k = 0; k < numDim && k < MAX_DIMENSIONS; ++k)
    {
        midpoint(k) = searchPointsFixed[bestIdx].midpoint[k];
        spread(k) = searchPointsFixed[bestIdx].spread[k];
    }

    return bestGMP;
}

/**
 * Ejecuta la fase completa de optimizacion de midpoint.
 * VERSION OPTIMIZADA.
 */
static int optimizeMidpointPhase(
    const Eigen::MatrixXd &legislatorCoords,
    int rollCallIndex,
    Eigen::VectorXd &midpoint,
    Eigen::VectorXd &spread,
    const VoteMatrix &votes,
    const Eigen::VectorXd &weights,
    const NormalCDF &normalCDF,
    const RollCallOptimizerConfig &config,
    double &initialGMP)
{
    std::vector<double> gmpHistory(config.numInnerIterations, 0.0);
    int iterations = 0;

    for (int iter = 0; iter < config.numInnerIterations; ++iter)
    {
        int validPoints = 0;
        double gmp = optimizeMidpointIteration(
            legislatorCoords, rollCallIndex, midpoint, spread,
            votes, weights, normalCDF, config, validPoints);

        gmpHistory[iter] = gmp;
        iterations = iter + 1;

        // Guardar GMP inicial
        if (iter == 0)
        {
            initialGMP = gmp;
        }

        // Verificar convergencia despues de la 3ra iteracion
        if (iter >= 2)
        {
            double improvement = gmpHistory[iter] - gmpHistory[iter - 1];
            if (improvement <= config.convergenceTol)
            {
                break;
            }
        }
    }

    // OPTIMIZACION: Eliminada llamada redundante a computeRollCallDerivativesOptimized
    // (resultado no se usaba)

    return iterations;
}

// Implementacion principal
RollCallOptimizationResult optimizeRollCall(
    const Eigen::MatrixXd &legislatorCoords,
    int rollCallIndex,
    const Eigen::VectorXd &initialMidpoint,
    const Eigen::VectorXd &initialSpread,
    const VoteMatrix &votes,
    const Eigen::VectorXd &weights,
    const NormalCDF &normalCDF,
    const RollCallOptimizerConfig &config)
{
    const int numDim = static_cast<int>(initialMidpoint.size());

    // Validaciones
    if (initialSpread.size() != numDim)
    {
        throw std::invalid_argument("Dimension de spread inconsistente");
    }
    if (weights.size() != numDim + 1)
    {
        throw std::invalid_argument("Vector de pesos debe tener NS+1 elementos");
    }

    // Inicializar resultado
    RollCallOptimizationResult result(numDim);
    result.midpoint = initialMidpoint;
    result.spread = initialSpread;

    // Calcular log-likelihood inicial
    auto initResult = computeRollCallDerivatives(
        legislatorCoords, rollCallIndex, result.midpoint, result.spread,
        votes, weights, normalCDF);

    result.initialGMP = initResult.geometricMeanProb;
    result.totalVotes = initResult.totalVotes;

    if (result.totalVotes == 0)
    {
        // Sin votos validos, retornar sin optimizar
        return result;
    }

    // Loop principal JJJJ (ciclos DYN + ZMID)
    int totalSpreadIter = 0;
    int totalMidpointIter = 0;

    for (int cycle = 0; cycle < config.numOuterIterations; ++cycle)
    {
        // FASE A: Optimizar spread (DYN)
        double spreadInitGMP = 0.0;
        int spreadIter = optimizeSpreadPhase(
            legislatorCoords, rollCallIndex, result.midpoint, result.spread,
            votes, weights, normalCDF, config, spreadInitGMP);
        totalSpreadIter += spreadIter;

        // FASE B: Optimizar midpoint (ZMID)
        double midpointInitGMP = 0.0;
        int midpointIter = optimizeMidpointPhase(
            legislatorCoords, rollCallIndex, result.midpoint, result.spread,
            votes, weights, normalCDF, config, midpointInitGMP);
        totalMidpointIter += midpointIter;
    }

    // Calcular resultado final
    auto finalResult = computeRollCallDerivatives(
        legislatorCoords, rollCallIndex, result.midpoint, result.spread,
        votes, weights, normalCDF);

    result.logLikelihood = finalResult.logLikelihood;
    result.geometricMeanProb = finalResult.geometricMeanProb;
    result.correctClassified = finalResult.correctClassified;
    result.spreadIterations = totalSpreadIter;
    result.midpointIterations = totalMidpointIter;
    result.totalIterations = totalSpreadIter + totalMidpointIter;
    result.converged = (result.geometricMeanProb >= result.initialGMP);

    return result;
}

// Version con RollCallParameters
RollCallOptimizationResult optimizeRollCall(
    const Eigen::MatrixXd &legislatorCoords,
    int rollCallIndex,
    const RollCallParameters &initialParams,
    const VoteMatrix &votes,
    const Eigen::VectorXd &weights,
    const NormalCDF &normalCDF,
    const RollCallOptimizerConfig &config)
{
    return optimizeRollCall(
        legislatorCoords,
        rollCallIndex,
        initialParams.midpoint,
        initialParams.spread,
        votes,
        weights,
        normalCDF,
        config);
}

/**
 * @brief Implementacion del optimizador de parametros de roll calls (RCINT2).
 */

#include "rollcall_optimizer.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>

// ============================================================================
// Estructuras auxiliares internas
// ============================================================================

/**
 * @brief Estado de un punto en la busqueda lineal.
 * YGAMMA, YGMP, YLOG en Fortran.
 */
struct SearchPoint
{
    Eigen::VectorXd midpoint; // YGAMMA(KK, 1:NS)
    Eigen::VectorXd spread;   // YGAMMA(KK, NS+1:2*NS)
    double gmp;               // YGMP(KK)
    double logLikelihood;     // YLOG(KK)
    int index;                // LLL(KK), para ordenamiento

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

/**
 * @brief Estado guardado para backtracking.
 * DZSAVE en Fortran.
 */
struct SavedState
{
    Eigen::VectorXd midpoint; // DZSAVE(1:NS)
    Eigen::VectorXd spread;   // DZSAVE(NS+1:2*NS)

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

// ============================================================================
// Funciones auxiliares
// ============================================================================

/**
 * @brief Protege spread contra valores muy pequenios.
 */
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
 * @brief Verifica si spread tiene valores protegidos.
 *
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
 * @brief Calcula el tamanio de paso normalizado.
 *
 * @param gradientNormSquared Norma al cuadrado del gradiente (SUMB)
 * @param stepUnit Unidad base (0.01)
 * @return Tamanio de paso
 */
static double computeStepSize(double gradientNormSquared, double stepUnit)
{
    if (gradientNormSquared < 1e-20)
    {
        return stepUnit; // Evitar division por cero
    }
    return stepUnit / std::sqrt(gradientNormSquared);
}

/**
 * @brief Verifica si todas las derivadas son esencialmente cero.
 */
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
 * @brief Proyecta midpoint a la superficie de la hiperesfera unitaria.
 *
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
 * @brief Encuentra el indice del punto con mayor GMP.
 *
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

// ============================================================================
// Fase A: Optimizacion de Spread (DYN)
// ============================================================================

/**
 * @brief Ejecuta una iteracion de optimizacion de spread.
 *
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

    // Guardar estado actual (DZSAVE)
    SavedState saved(numDim);
    saved.save(midpoint, spread);

    // Proteger spread contra valores extremos
    protectSpread(spread, config.minSpread, config.defaultSpread);
    saved.spread = spread;

    // Calcular derivadas con PROLLC2
    auto derivResult = computeRollCallDerivatives(
        legislatorCoords, rollCallIndex, midpoint, spread,
        votes, weights, normalCDF);

    int numVotes = derivResult.totalVotes;
    if (numVotes == 0)
    {
        exitEarly = true;
        return 0.0;
    }

    double currentGMP = derivResult.geometricMeanProb;

    // Normalizar gradiente (dividir por KRC)
    Eigen::VectorXd gradSpread = derivResult.spreadDerivatives / static_cast<double>(numVotes);

    // Verificar derivadas nulas
    if (areDerivativesZero(gradSpread, config.derivativeTol))
    {
        exitEarly = true;
        return currentGMP;
    }

    // Calcular norma del gradiente y tamanio de paso
    double gradNormSq = gradSpread.squaredNorm();
    double stepSize = computeStepSize(gradNormSq, config.stepUnit);

    // Busqueda lineal: evaluar NINC puntos
    std::vector<SearchPoint> searchPoints(config.numSearchPoints, SearchPoint(numDim));
    double stepAccum = 0.0;

    for (int kk = 0; kk < config.numSearchPoints; ++kk)
    {
        // Actualizar spread en direccion del gradiente negativo
        for (int k = 0; k < numDim; ++k)
        {
            spread(k) = saved.spread(k) - stepAccum * gradSpread(k);
        }

        // Evaluar en este punto
        auto evalResult = computeRollCallDerivatives(
            legislatorCoords, rollCallIndex, midpoint, spread,
            votes, weights, normalCDF);

        // Guardar punto
        searchPoints[kk].midpoint = midpoint;
        searchPoints[kk].spread = spread;
        searchPoints[kk].gmp = evalResult.geometricMeanProb;
        searchPoints[kk].logLikelihood = evalResult.logLikelihood;
        searchPoints[kk].index = kk;

        stepAccum += stepSize;
    }

    // Encontrar el mejor punto
    int bestIdx = findBestPoint(searchPoints, config.numSearchPoints);

    // Actualizar parametros al mejor punto
    midpoint = searchPoints[bestIdx].midpoint;
    spread = searchPoints[bestIdx].spread;

    // Proteger spread despues de actualizacion
    if (checkAndProtectSpread(spread, config.minSpread, config.defaultSpread))
    {
        exitEarly = true;
    }

    return searchPoints[bestIdx].gmp;
}

/**
 * @brief Ejecuta la fase completa de optimizacion de spread.
 *
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

    // Verificacion post-optimizacion
    auto finalResult = computeRollCallDerivatives(
        legislatorCoords, rollCallIndex, midpoint, spread,
        votes, weights, normalCDF);

    return iterations;
}

// ============================================================================
// Fase B: Optimizacion de Midpoint (ZMID)
// ============================================================================

/**
 * @brief Ejecuta una iteracion de optimizacion de midpoint.
 *
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

    // Guardar estado actual
    SavedState saved(numDim);
    saved.save(midpoint, spread);

    // Calcular derivadas
    auto derivResult = computeRollCallDerivatives(
        legislatorCoords, rollCallIndex, midpoint, spread,
        votes, weights, normalCDF);

    int numVotes = derivResult.totalVotes;
    if (numVotes == 0)
    {
        validPoints = 0;
        return 0.0;
    }

    // Normalizar gradiente
    Eigen::VectorXd gradMidpoint = derivResult.midpointDerivatives / static_cast<double>(numVotes);

    // Calcular norma y tamanio de paso
    double gradNormSq = gradMidpoint.squaredNorm();
    double stepSize = computeStepSize(gradNormSq, config.stepUnit);

    // Busqueda lineal con restriccion esferica
    std::vector<SearchPoint> searchPoints(config.numSearchPoints, SearchPoint(numDim));
    double stepAccum = 0.0;
    validPoints = 0;

    for (int kk = 0; kk < config.numSearchPoints; ++kk)
    {
        // Actualizar midpoint en direccion del gradiente negativo
        for (int k = 0; k < numDim; ++k)
        {
            midpoint(k) = saved.midpoint(k) - stepAccum * gradMidpoint(k);
        }

        // Verificar restriccion de hiperesfera unitaria
        bool projected = projectToUnitSphere(midpoint);

        // Evaluar en este punto
        auto evalResult = computeRollCallDerivatives(
            legislatorCoords, rollCallIndex, midpoint, spread,
            votes, weights, normalCDF);

        // Guardar punto
        searchPoints[kk].midpoint = midpoint;
        searchPoints[kk].spread = spread;
        searchPoints[kk].gmp = evalResult.geometricMeanProb;
        searchPoints[kk].logLikelihood = evalResult.logLikelihood;
        searchPoints[kk].index = kk;

        validPoints = kk + 1;

        // Si se proyecto, terminar busqueda (como en Fortran GO TO 2113)
        if (projected)
        {
            break;
        }

        stepAccum += stepSize;
    }

    // Encontrar el mejor punto
    int bestIdx = findBestPoint(searchPoints, validPoints);

    // Actualizar parametros
    midpoint = searchPoints[bestIdx].midpoint;
    spread = searchPoints[bestIdx].spread;

    return searchPoints[bestIdx].gmp;
}

/**
 * @brief Ejecuta la fase completa de optimizacion de midpoint.
 *
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

    // Verificacion post-optimizacion
    auto finalResult = computeRollCallDerivatives(
        legislatorCoords, rollCallIndex, midpoint, spread,
        votes, weights, normalCDF);

    return iterations;
}

// ============================================================================
// Implementacion principal
// ============================================================================

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

    // Bloque 2: Calcular log-likelihood inicial
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

    // Bloque 3: Loop principal JJJJ (ciclos DYN + ZMID)
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

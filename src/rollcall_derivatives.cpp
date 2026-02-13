/**
 * Implementacion del calculo de derivadas para parametros de roll calls.
 */

#include "rollcall_derivatives.hpp"
#include <cmath>
#include <stdexcept>

// Estructuras auxiliares para calculos intermedios

/**
 * Almacena distancias cuadradas y lineales por dimension.
 * DYES, DNO, DYES1, DNO1 en Fortran.
 */
struct DimensionalDistances
{
    Eigen::VectorXd yesSquared; // DYES: (x - zmid + dyn)^2
    Eigen::VectorXd noSquared;  // DNO:  (x - zmid - dyn)^2
    Eigen::VectorXd yesLinear;  // DYES1: (x - zmid + dyn)
    Eigen::VectorXd noLinear;   // DNO1:  (x - zmid - dyn)

    explicit DimensionalDistances(int numDim)
        : yesSquared(Eigen::VectorXd::Zero(numDim)),
          noSquared(Eigen::VectorXd::Zero(numDim)),
          yesLinear(Eigen::VectorXd::Zero(numDim)),
          noLinear(Eigen::VectorXd::Zero(numDim))
    {
    }
};

/**
 * Almacena utilidades y terminos derivativos para un voto.
 * DC, DB, DCC, DBB, DCC1, DBB1, XCC en Fortran.
 */
struct VoteUtilities
{
    double DC;                     // Utilidad del voto correcto
    double DB;                     // Utilidad del voto incorrecto
    Eigen::VectorXd DCC;           // Distancias cuadradas para voto correcto
    Eigen::VectorXd DBB;           // Distancias cuadradas para voto incorrecto
    Eigen::VectorXd DCC1_weighted; // Distancias lineales ponderadas (correcto)
    Eigen::VectorXd DBB1_weighted; // Distancias lineales ponderadas (incorrecto)
    double XCC;                    // +1 si voto Si, -1 si voto No

    explicit VoteUtilities(int numDim)
        : DC(0.0),
          DB(0.0),
          DCC(Eigen::VectorXd::Zero(numDim)),
          DBB(Eigen::VectorXd::Zero(numDim)),
          DCC1_weighted(Eigen::VectorXd::Zero(numDim)),
          DBB1_weighted(Eigen::VectorXd::Zero(numDim)),
          XCC(0.0)
    {
    }
};

// Funciones auxiliares

/**
 *  Calcula distancias cuadradas y lineales para un legislador.
 */
static DimensionalDistances computeDistances(
    const Eigen::VectorXd &legislatorCoord,
    const Eigen::VectorXd &midpoint,
    const Eigen::VectorXd &spread)
{
    const int numDim = static_cast<int>(midpoint.size());
    DimensionalDistances dist(numDim);

    for (int k = 0; k < numDim; ++k)
    {
        // DYES1 = x - zmid + dyn (distancia al punto zmid - dyn)
        double diffYes = legislatorCoord(k) - midpoint(k) + spread(k);
        dist.yesLinear(k) = diffYes;
        dist.yesSquared(k) = diffYes * diffYes;

        // DNO1 = x - zmid - dyn (distancia al punto zmid + dyn)
        double diffNo = legislatorCoord(k) - midpoint(k) - spread(k);
        dist.noLinear(k) = diffNo;
        dist.noSquared(k) = diffNo * diffNo;
    }

    return dist;
}

/**
 * Calcula utilidades y terminos derivativos para un voto.
 *
 * @param dist Distancias precalculadas
 * @param weights Vector de pesos [w1,...,wNS,beta]
 * @param votedYes true si el legislador voto Si
 * @return Estructura con utilidades y terminos para derivadas
 */
static VoteUtilities computeUtilities(
    const DimensionalDistances &dist,
    const Eigen::VectorXd &weights,
    bool votedYes)
{
    const int numDim = static_cast<int>(dist.yesSquared.size());
    VoteUtilities util(numDim);

    util.DC = 0.0;
    util.DB = 0.0;

    for (int k = 0; k < numDim; ++k)
    {
        double wk_sq = weights(k) * weights(k);

        if (votedYes)
        {
            // DC usa DYES (voto correcto), DB usa DNO (voto incorrecto)
            util.DC += -wk_sq * dist.yesSquared(k);
            util.DB += -wk_sq * dist.noSquared(k);
            util.DCC(k) = dist.yesSquared(k);
            util.DBB(k) = dist.noSquared(k);
            util.DCC1_weighted(k) = dist.yesLinear(k) * wk_sq;
            util.DBB1_weighted(k) = dist.noLinear(k) * wk_sq;
        }
        else
        {
            // DC usa DNO (voto correcto), DB usa DYES (voto incorrecto)
            util.DC += -wk_sq * dist.noSquared(k);
            util.DB += -wk_sq * dist.yesSquared(k);
            util.DCC(k) = dist.noSquared(k);
            util.DBB(k) = dist.yesSquared(k);
            util.DCC1_weighted(k) = dist.noLinear(k) * wk_sq;
            util.DBB1_weighted(k) = dist.yesLinear(k) * wk_sq;
        }
    }

    util.XCC = votedYes ? +1.0 : -1.0;

    return util;
}

/**
 * Actualiza estadisticas de clasificacion.
 * Clasificacion correcta si |DC| <= |DB|
 * Matriz de confusion basada en XCC y clasificacion
 */
static void updateClassificationStats(
    RollCallDerivativesResult &result,
    double DC,
    double DB,
    double XCC,
    double ZS)
{
    bool correctlyClassified = (std::abs(DC) <= std::abs(DB));

    if (correctlyClassified)
    {
        result.correctClassified++;
    }

    // Matriz de confusion
    if (correctlyClassified && XCC > 0.0)
    {
        result.truePositives++; // KLASSYY
    }
    if (!correctlyClassified && XCC > 0.0)
    {
        result.falseNegatives++; // KLASSNY
    }
    if (!correctlyClassified && XCC < 0.0)
    {
        result.falsePositives++; // KLASSYN
    }
    if (correctlyClassified && XCC < 0.0)
    {
        result.trueNegatives++; // KLASSNN
    }

    // Clasificacion alternativa por signo de ZS
    if (ZS > 0.0)
    {
        result.positiveZS++; // KLASS2
    }
}

// Implementacion principal
RollCallDerivativesResult computeRollCallDerivatives(
    const Eigen::MatrixXd &legislatorCoords,
    int rollCallIndex,
    const Eigen::VectorXd &midpoint,
    const Eigen::VectorXd &spread,
    const VoteMatrix &votes,
    const Eigen::VectorXd &weights,
    const NormalCDF &normalCDF)
{
    // Validacion de dimensiones
    const int numLegislators = static_cast<int>(legislatorCoords.rows());
    const int numDimensions = static_cast<int>(legislatorCoords.cols());

    if (midpoint.size() != numDimensions)
    {
        throw std::invalid_argument(
            "Dimension de midpoint inconsistente con coordenadas");
    }
    if (spread.size() != numDimensions)
    {
        throw std::invalid_argument(
            "Dimension de spread inconsistente con coordenadas");
    }
    if (weights.size() != numDimensions + 1)
    {
        throw std::invalid_argument(
            "Vector de pesos debe tener NS+1 elementos");
    }
    if (rollCallIndex < 0 ||
        rollCallIndex >= static_cast<int>(votes.getNumRollCalls()))
    {
        throw std::out_of_range("Indice de roll call fuera de rango");
    }

    // Inicializacion
    RollCallDerivativesResult result(numDimensions);

    // Beta = WEIGHT(NS+1)
    const double beta = weights(numDimensions);

    // Loop principal sobre legisladores
    for (int i = 0; i < numLegislators; ++i)
    {
        // Calcular distancias
        Eigen::VectorXd legCoord = legislatorCoords.row(i).transpose();
        DimensionalDistances dist = computeDistances(legCoord, midpoint, spread);

        // Verificar si hay dato valido
        if (votes.isMissing(i, rollCallIndex))
        {
            continue; // Saltar votos faltantes
        }

        result.totalVotes++; // KTOT++

        // Calcular utilidades segun voto
        bool votedYes = votes.getVote(i, rollCallIndex);
        VoteUtilities util = computeUtilities(dist, weights, votedYes);

        // Calcular ZS
        // ZS = WEIGHT(NS+1) * (EXP(DC) - EXP(DB))
        double expDC = std::exp(util.DC);
        double expDB = std::exp(util.DB);
        double ZS = beta * (expDC - expDB);

        // Actualizar estadisticas de clasificacion
        updateClassificationStats(result, util.DC, util.DB, util.XCC, ZS);

        // Lookup en tabla CDF
        double cdfValue = normalCDF.cdf(ZS);
        double logCdfValue = normalCDF.logCdf(ZS);

        // Acumular log-likelihood
        result.logLikelihood += logCdfValue;

        // Calcular derivadas
        // ZGAUSS = exp(-(ZS*ZS)/2.0)
        double zgauss = std::exp(-(ZS * ZS) / 2.0);

        // Ratio phi(ZS)/Phi(ZS) para derivadas
        // Evitar division por cero
        double ratio = 0.0;
        if (cdfValue > 1e-300)
        {
            ratio = zgauss / cdfValue;
        }

        // Derivadas respecto a midpoint y spread
        for (int k = 0; k < numDimensions; ++k)
        {
            // ZDERV(K) = ZDERV(K) + (ZGAUSS/ZDISTF) *
            //            (-DCC1(K)*EXP(DC) + DBB1(K)*EXP(DB))
            result.midpointDerivatives(k) += ratio *
                                             (-util.DCC1_weighted(k) * expDC + util.DBB1_weighted(k) * expDB);

            // DDERV(K) = DDERV(K) + XCC * (ZGAUSS/ZDISTF) *
            //            (DCC1(K)*EXP(DC) + DBB1(K)*EXP(DB))
            result.spreadDerivatives(k) += util.XCC * ratio *
                                           (util.DCC1_weighted(k) * expDC + util.DBB1_weighted(k) * expDB);
        }
    }

    // Calcular probabilidad geometrica media
    // GMP = EXP(XPLOG/FLOAT(KTOT))
    if (result.totalVotes > 0)
    {
        result.geometricMeanProb = std::exp(
            result.logLikelihood / static_cast<double>(result.totalVotes));
    }

    return result;
}

// Version con RollCallParameters
RollCallDerivativesResult computeRollCallDerivatives(
    const Eigen::MatrixXd &legislatorCoords,
    int rollCallIndex,
    const RollCallParameters &rcParams,
    const VoteMatrix &votes,
    const Eigen::VectorXd &weights,
    const NormalCDF &normalCDF)
{
    return computeRollCallDerivatives(
        legislatorCoords,
        rollCallIndex,
        rcParams.midpoint,
        rcParams.spread,
        votes,
        weights,
        normalCDF);
}

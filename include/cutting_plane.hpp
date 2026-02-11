/**
 * @file cutting_plane.hpp
 * @brief Algoritmos de plano de corte para DW-NOMINATE (SEARCH/CUTPLANE).
 *
 * Subrutinas de plano de corte del codigo Fortran.
 *
 * Funciones implementadas:
 * - refineCuttingPlane(): SEARCH
 */

#ifndef CUTTING_PLANE_HPP
#define CUTTING_PLANE_HPP

#include "cutting_point.hpp"
#include "sort_utils.hpp"
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <vector>

/**
 * @brief Resultado del refinamiento del plano de corte (SEARCH).
 *
 * Contiene el vector normal optimizado, punto de corte, polaridad
 * y estadisticas de clasificacion.
 */
struct SearchResult
{
    // Vector normal optimizado (equivalente a ZVEC(JX,:) de salida)
    Eigen::VectorXd normalVector;

    // Punto de corte optimo (equivalente a WS(JX) de salida)
    double cuttingPoint;

    // Polaridad del corte (KCUT, LCUT)
    CuttingPolarity polarity;

    // Conteos de clasificacion del mejor resultado
    ClassificationCounts counts;

    // Numero de errores de la mejor solucion (KITTY1)
    int errors;

    // Total de votos clasificados (KITTY2)
    int totalClassified;

    // Proyecciones finales de legisladores (XXY)
    std::vector<double> projections;

    // Numero de la iteracion que produjo el mejor resultado
    int bestIteration;

    // Flag indicando si se encontro clasificacion perfecta
    bool isPerfectClassification;

    SearchResult()
        : cuttingPoint(0.0),
          errors(0),
          totalClassified(0),
          bestIteration(0),
          isPerfectClassification(false) {}

    explicit SearchResult(int numDimensions)
        : normalVector(Eigen::VectorXd::Zero(numDimensions)),
          cuttingPoint(0.0),
          errors(0),
          totalClassified(0),
          bestIteration(0),
          isPerfectClassification(false) {}
};

/**
 * @brief Estado de una iteracion de SEARCH para almacenamiento interno.
 *
 * Corresponde a las variables UUU, FV1, FV2, KKKCUT, LLLCUT del Fortran.
 */
struct SearchIterationState
{
    Eigen::VectorXd normalVector; // UUU(IJL,:): Vector normal de esta iteracion
    double errorCount;            // FV1(IJL): Numero de errores como double
    double cuttingPoint;          // FV2(IJL): Punto de corte
    int kcut;                     // KKKCUT(IJL): Polaridad lado bajo
    int lcut;                     // LLLCUT(IJL): Polaridad lado alto
    int totalErrors;              // LLN(IJL): Total de errores como int
    ClassificationCounts counts;  // Conteos de clasificacion

    SearchIterationState()
        : errorCount(999.0),
          cuttingPoint(0.0),
          kcut(VoteCode::YES),
          lcut(VoteCode::NO),
          totalErrors(0) {}
};

/**
 * @brief Refina la orientacion del plano de corte para minimizar errores (SEARCH).
 *
 * @param legislatorCoords Coordenadas de legisladores (NP x NS) [XMAT]
 * @param votes Vector de votos para este roll call [LDATA(:,JX)]
 *              Codigos: 1=Si, 6=No, 0/9=Ausente
 * @param initialNormal Vector normal inicial [ZVEC(JX,:)]
 * @param initialCutPoint Punto de corte inicial [WS(JX)]
 * @param initialPolarity Polaridad inicial [KCUT, LCUT]
 * @param maxIterations Numero maximo de iteraciones [NCUT], default=25
 * @return SearchResult con vector optimizado y estadisticas
 *
 * @note El vector normal de entrada se usa como punto de partida.
 * @note Los votos con codigo 0 se tratan como 9 (ausente).
 */
SearchResult refineCuttingPlane(
    const Eigen::MatrixXd &legislatorCoords,
    const std::vector<int> &votes,
    const Eigen::VectorXd &initialNormal,
    double initialCutPoint,
    const CuttingPolarity &initialPolarity,
    int maxIterations = 25);

/**
 * @brief Version que actualiza matrices in-place
 *
 * @param legislatorCoords Coordenadas de legisladores (NP x NS)
 * @param votes Vector de votos para este roll call
 * @param normalVector Vector normal (entrada/salida)
 * @param cuttingPoint Punto de corte (entrada/salida)
 * @param polarity Polaridad (entrada/salida)
 * @param projections Proyecciones de salida (se redimensiona)
 * @param maxIterations Numero maximo de iteraciones
 * @param outErrors Numero de errores de salida [KITTY1]
 * @param outTotalClassified Total clasificados de salida [KITTY2]
 * @return true si se encontro clasificacion perfecta, false en caso contrario
 */
bool refineCuttingPlaneInPlace(
    const Eigen::MatrixXd &legislatorCoords,
    const std::vector<int> &votes,
    Eigen::VectorXd &normalVector,
    double &cuttingPoint,
    CuttingPolarity &polarity,
    std::vector<double> &projections,
    int maxIterations,
    int &outErrors,
    int &outTotalClassified);

#endif // CUTTING_PLANE_HPP

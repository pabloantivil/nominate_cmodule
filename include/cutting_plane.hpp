/**
 * @file cutting_plane.hpp
 * Algoritmos de plano de corte para DW-NOMINATE (SEARCH/CUTPLANE).
 */

#ifndef CUTTING_PLANE_HPP
#define CUTTING_PLANE_HPP

#include "cutting_point.hpp"
#include "sort_utils.hpp"
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <vector>

// Flag global para debug de SEARCH (definido en cutting_plane.cpp)
extern bool g_searchDebug;

/**
 * Resultado del refinamiento del plano de corte (SEARCH
 * Contiene el vector normal optimizado, punto de corte, polaridad y estadisticas de clasificacion.
 */
struct SearchResult
{
    Eigen::VectorXd normalVector;    // Vector normal optimizado (equivalente a ZVEC(JX,:) de salida)
    double cuttingPoint;             // Punto de corte optimo (equivalente a WS(JX) de salida)
    CuttingPolarity polarity;        // Polaridad del corte (KCUT, LCUT)
    ClassificationCounts counts;     // Conteos de clasificacion del mejor resultado
    int errors;                      // Numero de errores de la mejor solucion (KITTY1)
    int totalClassified;             // Total de votos clasificados (KITTY2)
    std::vector<double> projections; // Proyecciones finales de legisladores (XXY)
    int bestIteration;               // Numero de la iteracion que produjo el mejor resultado
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

// Estado de una iteracion de SEARCH para almacenamiento interno.
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
 * Refina la orientacion del plano de corte para minimizar errores (SEARCH).
 * @param legislatorCoords Coordenadas de legisladores (NP x NS) [XMAT]
 * @param votes Vector de votos para este roll call [LDATA(:,JX)]
 *              Codigos: 1=Si, 6=No, 0/9=Ausente
 * @param initialNormal Vector normal inicial [ZVEC(JX,:)]
 * @param initialCutPoint Punto de corte inicial [WS(JX)]
 * @param initialPolarity Polaridad inicial [KCUT, LCUT]
 * @param maxIterations Numero maximo de iteraciones [NCUT], default=25
 * @return SearchResult con vector optimizado y estadisticas
 */
SearchResult refineCuttingPlane(
    const Eigen::MatrixXd &legislatorCoords,
    const std::vector<int> &votes,
    const Eigen::VectorXd &initialNormal,
    double initialCutPoint,
    const CuttingPolarity &initialPolarity,
    int maxIterations = 25);

// CUTPLANE - Orquestador de clasificacion inicial de todas las votaciones
// Estadisticas de proyecciones para una votacion.
struct ProjectionStats
{
    double meanYes; // SUMYES: media de proyecciones de votantes "Si"
    double meanNo;  // SUMNO: media de proyecciones de votantes "No"
    int countYes;   // KSUMYES: numero de votantes "Si"
    int countNo;    // KSUMNO: numero de votantes "No"

    ProjectionStats() : meanYes(0.0), meanNo(0.0), countYes(0), countNo(0) {}
};


// Resultado de clasificacion para una votacion individual.
struct RollCallClassification
{
    CuttingPolarity polarity;          // Polaridad de la votacion (KCUT, LCUT)
    double cuttingPoint;               // Punto de corte optimo (WS(JX))
    std::vector<int> legislatorErrors; // Errores por legislador para esta votacion
    ClassificationCounts counts;       // Conteos de clasificacion
    ProjectionStats projStats;         // Estadisticas de proyecciones
    std::vector<double> projections;   // Proyecciones finales de legisladores
    bool searchPerformed;              // Flag indicando si se ejecuto SEARCH
    int totalErrors;                   // Numero de errores de clasificacion
    int totalClassified;               // Total de votos clasificados (excluye ausentes)

    RollCallClassification()
        : cuttingPoint(0.0),
          searchPerformed(false),
          totalErrors(0),
          totalClassified(0) {}
};


// Resultado completo de CUTPLANE para todas las votaciones.
struct CutplaneResult
{
    std::vector<CuttingPolarity> polarities;             // Polaridades de todas las votaciones (MCUTS)
    std::vector<double> cuttingPoints;                   // Puntos de corte de todas las votaciones (WS(:))
    std::vector<RollCallClassification> rollCallResults; // Clasificacion detallada por roll call
    int totalClassified;                                 // Total de votos clasificados (KT)
    int totalErrors;                                     // Total de errores de clasificacion (KTT)
    double errorRate;                                    // Tasa de error global (XERROR = KTT/KT)
    double accuracy;                                     // Tasa de acierto global (YERROR = 1 - XERROR)
    int numLegislators;                                  // Numero de legisladores (NP)
    int numRollCalls;                                    // Numero de votaciones (NRCALL)
    int numDimensions;                                   // Numero de dimensiones (NS)

    CutplaneResult()
        : totalClassified(0),
          totalErrors(0),
          errorRate(0.0),
          accuracy(1.0),
          numLegislators(0),
          numRollCalls(0),
          numDimensions(0) {}

    // Obtiene la matriz de errores LERROR(i,jx) completa.
    Eigen::MatrixXi getLegislatorErrorMatrix() const
    {
        Eigen::MatrixXi lerror = Eigen::MatrixXi::Zero(numLegislators, numRollCalls);
        for (int jx = 0; jx < numRollCalls; ++jx)
        {
            for (int i = 0; i < numLegislators; ++i)
            {
                lerror(i, jx) = rollCallResults[jx].legislatorErrors[i];
            }
        }
        return lerror;
    }

    /**
     * Obtiene errores totales por legislador.
     * @return Vector de tamano NP con total de errores por legislador
     */
    std::vector<int> getErrorsByLegislator() const
    {
        std::vector<int> errors(numLegislators, 0);
        for (int jx = 0; jx < numRollCalls; ++jx)
        {
            for (int i = 0; i < numLegislators; ++i)
            {
                errors[i] += rollCallResults[jx].legislatorErrors[i];
            }
        }
        return errors;
    }
};

/**
 * Clasificacion inicial de todas las votaciones (CUTPLANE).
 * @param legislatorCoords Coordenadas de legisladores (NP x NS) [XMAT]
 * @param normalVectors Vectores normales de roll calls (NRCALL x NS) [ZVEC]
 * @param voteMatrix Matriz de votos (NP x NRCALL) [LDATA]
 *                   Codigos: 1=Si, 6=No, 0=Ausente
 * @param searchEnabled Si true, ejecutar SEARCH cuando hay errores [IFIXX != 0]
 * @return CutplaneResult con clasificacion completa
 */
CutplaneResult findAllCuttingPlanes(
    const Eigen::MatrixXd &legislatorCoords,
    Eigen::MatrixXd &normalVectors,
    const Eigen::MatrixXi &voteMatrix,
    bool searchEnabled = true);

/**
 * Clasificacion de una votacion individual.
 * @param legislatorCoords Coordenadas de legisladores (NP x NS)
 * @param normalVector Vector normal de esta votacion (NS) - entrada/salida
 * @param votes Vector de votos para esta votacion (NP)
 * @param searchEnabled Si true, ejecutar SEARCH cuando hay errores
 * @return RollCallClassification con clasificacion de esta votacion
 */
RollCallClassification classifyRollCall(
    const Eigen::MatrixXd &legislatorCoords,
    Eigen::VectorXd &normalVector,
    const std::vector<int> &votes,
    bool searchEnabled = true);

#endif // CUTTING_PLANE_HPP

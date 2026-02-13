/**
 * @file cutting_point.cpp
 * Implementacion de algoritmos de punto de corte optimo (JAN1PT/JAN11PT).
 *
 * Funciones traducidas:
 * - JAN1PT -> findCuttingPoint1D()
 * - JAN11PT -> findCuttingPoint1DFixedPolarity()
 *
 */

#include "cutting_point.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace
{

    /**
     * Estructura interna para un corte candidato.
     */
    struct CutCandidate
    {
        double position;   // Y(I): Posicion del corte
        double errorRate;  // Z(I): Tasa de error
        int correctLow;    // LV(I): Correctos lado bajo
        int errorsLow;     // LE(I): Errores lado bajo
        int correctHigh;   // LVB(I): Correctos lado alto
        int errorsHigh;    // LEB(I): Errores lado alto
        int originalIndex; // LJEP(I): Indice original del corte

        CutCandidate()
            : position(0.0), errorRate(999.0),
              correctLow(0), errorsLow(0), correctHigh(0), errorsHigh(0),
              originalIndex(0) {}
    };

    /**
     * Resultado de evaluacion para una polaridad.
     *
     * Almacena el mejor corte encontrado para una configuracion KCUT/LCUT.
     */
    struct PolarityResult
    {
        double errorRate;   // AA: Tasa de error del mejor corte
        double cutPosition; // AB: Posicion del mejor corte
        int correctLow;     // LA: JCL
        int errorsLow;      // LB: JEL
        int correctHigh;    // LC: JCH
        int errorsHigh;     // LD: JEH

        PolarityResult()
            : errorRate(999.0), cutPosition(0.0),
              correctLow(0), errorsLow(0), correctHigh(0), errorsHigh(0) {}
    };

    /**
     * Genera puntos de corte candidatos.
     *
     * @param projections Proyecciones ordenadas
     * @return Vector de posiciones de corte candidatas
     */
    std::vector<double> generateCutPoints(const std::vector<double> &projections)
    {
        const size_t np = projections.size();
        std::vector<double> cutPoints(np);

        if (np == 0)
            return cutPoints;

        // Primer corte: 0.001 a la izquierda del primer punto
        cutPoints[0] = projections[0] - 0.001;

        // Cortes intermedios: punto medio entre proyecciones consecutivas
        for (size_t i = 1; i < np; ++i)
        {
            cutPoints[i] = (projections[i] + projections[i - 1]) / 2.0;
        }

        return cutPoints;
    }

    /**
     * Cuenta clasificacion para todos los cortes con una polaridad dada.
     *
     * @param projections Proyecciones ordenadas
     * @param votes Votos reordenados (1=Si, 6=No, 9=Ausente)
     * @param cutPoints Posiciones de corte candidatas
     * @param kcut Voto esperado en lado bajo
     * @param lcut Voto esperado en lado alto
     * @return Vector de candidatos evaluados
     */
    std::vector<CutCandidate> countClassificationForPolarity(
        const std::vector<double> &projections,
        const std::vector<int> &votes,
        const std::vector<double> &cutPoints,
        int kcut,
        int lcut)
    {
        const size_t np = projections.size();
        std::vector<CutCandidate> candidates(np);

        if (np == 0)
            return candidates;

        // Contadores incrementales
        int kSideVotes = 0;    // Correctos en lado bajo (K-Side Votes)
        int kSideErrors = 0;   // Errores en lado bajo (K-Side Errors)
        int lSideVotes = 0;    // Correctos en lado alto (L-Side Votes)
        int lSideErrors = 0;   // Errores en lado alto (L-Side Errors)
        bool firstPass = true; // Flag para primer corte (todos en lado alto)

        for (size_t i = 0; i < np; ++i)
        {
            candidates[i].position = cutPoints[i];
            candidates[i].originalIndex = static_cast<int>(i);

            if (firstPass)
            {
                // Primer corte: todos los legisladores estan en lado alto
                for (size_t j = i; j < np; ++j)
                {
                    int vote = votes[j];
                    if (vote == VoteCode::MISSING)
                        continue;

                    if (vote == lcut)
                    {
                        lSideVotes++; // Correcto en lado alto
                    }
                    else if (vote == kcut)
                    {
                        lSideErrors++; // Error en lado alto (deberia estar en bajo)
                    }
                }
                firstPass = false;
            }
            else
            {
                // Cortes posteriores: actualizar incrementalmente
                // El legislador i-1 pasa del lado alto al lado bajo
                int prevVote = votes[i - 1];
                if (prevVote != VoteCode::MISSING)
                {
                    if (prevVote == kcut)
                    {
                        // Era error en lado alto, ahora correcto en lado bajo
                        kSideVotes++;
                        lSideErrors--;
                    }
                    else if (prevVote == lcut)
                    {
                        // Era correcto en lado alto, ahora error en lado bajo
                        kSideErrors++;
                        lSideVotes--;
                    }
                }
            }

            // Almacenar conteos para este corte
            candidates[i].correctLow = kSideVotes;
            candidates[i].errorsLow = kSideErrors;
            candidates[i].correctHigh = lSideVotes;
            candidates[i].errorsHigh = lSideErrors;

            // Calcular tasa de error
            int total = kSideVotes + kSideErrors + lSideVotes + lSideErrors;
            if (total > 0)
            {
                candidates[i].errorRate =
                    static_cast<double>(kSideErrors + lSideErrors) / total;
            }
            else
            {
                candidates[i].errorRate = 999.0;
            }
        }

        return candidates;
    }

    /**
     * Encuentra el mejor corte entre candidatos, manejando empates.
     *
     * @param candidates Vector de candidatos evaluados
     * @return Mejor candidato seleccionado
     */
    CutCandidate findBestCutPoint(std::vector<CutCandidate> &candidates)
    {
        const size_t np = candidates.size();

        if (np == 0)
        {
            return CutCandidate();
        }

        // Ordenar por tasa de error (ascendente)
        std::vector<size_t> sortedIndices = argsort(
            std::vector<double>(np, 0.0)); // Placeholder

        // Crear vector de error rates para ordenar
        std::vector<double> errorRates(np);
        for (size_t i = 0; i < np; ++i)
        {
            errorRates[i] = candidates[i].errorRate;
        }
        sortedIndices = argsort(errorRates);

        // El mejor es el primero despues de ordenar
        size_t bestIdx = sortedIndices[0];
        double bestError = candidates[bestIdx].errorRate;

        // Buscar empates
        constexpr double TIE_TOLERANCE = 0.00001;
        constexpr size_t MAX_TIES = 100;

        std::vector<size_t> tiedIndices;
        tiedIndices.push_back(bestIdx);

        for (size_t i = 1; i < np && tiedIndices.size() < MAX_TIES; ++i)
        {
            size_t idx = sortedIndices[i];
            if (std::abs(candidates[idx].errorRate - bestError) <= TIE_TOLERANCE)
            {
                tiedIndices.push_back(idx);
            }
            else if (candidates[idx].errorRate > bestError)
            {
                break; // Los siguientes seran peores
            }
        }

        // Desempate: seleccionar el mas cercano al origen
        if (tiedIndices.size() == 1)
        {
            return candidates[tiedIndices[0]];
        }

        // Ordenar empates por valor absoluto de posicion
        std::sort(tiedIndices.begin(), tiedIndices.end(),
                  [&candidates](size_t a, size_t b)
                  {
                      return std::abs(candidates[a].position) <
                             std::abs(candidates[b].position);
                  });

        return candidates[tiedIndices[0]];
    }

    /**
     * Evalua todos los cortes para una polaridad.
     *
     * @param projections Proyecciones ordenadas
     * @param votes Votos reordenados
     * @param kcut Voto esperado en lado bajo
     * @param lcut Voto esperado en lado alto
     * @param outCandidates Candidatos evaluados (opcional, para modo ROTATION)
     * @return Resultado de la mejor clasificacion para esta polaridad
     */
    PolarityResult evaluatePolarityCuts(
        const std::vector<double> &projections,
        const std::vector<int> &votes,
        int kcut,
        int lcut,
        CuttingPointCandidates *outCandidates = nullptr)
    {
        PolarityResult result;

        if (projections.empty())
        {
            return result;
        }

        // Generar puntos de corte
        std::vector<double> cutPoints = generateCutPoints(projections);

        // Evaluar clasificacion para cada corte
        std::vector<CutCandidate> candidates =
            countClassificationForPolarity(projections, votes, cutPoints, kcut, lcut);

        // Almacenar candidatos intermedios si se solicita (JROTC=0)
        if (outCandidates != nullptr)
        {
            const size_t np = candidates.size();
            outCandidates->resize(np);
            for (size_t i = 0; i < np; ++i)
            {
                outCandidates->positions[i] = candidates[i].position;
                outCandidates->correctLow[i] = candidates[i].correctLow;
                outCandidates->errorsLow[i] = candidates[i].errorsLow;
                outCandidates->correctHigh[i] = candidates[i].correctHigh;
                outCandidates->errorsHigh[i] = candidates[i].errorsHigh;
            }
        }

        // Encontrar mejor corte
        CutCandidate best = findBestCutPoint(candidates);

        result.errorRate = best.errorRate;
        result.cutPosition = best.position;
        result.correctLow = best.correctLow;
        result.errorsLow = best.errorsLow;
        result.correctHigh = best.correctHigh;
        result.errorsHigh = best.errorsHigh;

        return result;
    }

    /**
     * Calcula centroides de cada grupo de clasificacion.
     *
     * @param projections Proyecciones ordenadas
     * @param votes Votos reordenados
     * @param originalIndices Mapeo a indices originales
     * @param legislatorCoords Coordenadas de legisladores
     * @param cutPosition Posicion del corte optimo
     * @param polarity Polaridad del corte
     * @param counts Conteos de clasificacion
     * @param numDimensions Numero de dimensiones
     * @return Centroides de cada grupo
     */
    GroupCentroids computeGroupCentroids(
        const std::vector<double> &projections,
        const std::vector<int> &votes,
        const std::vector<int> &originalIndices,
        const Eigen::MatrixXd &legislatorCoords,
        double cutPosition,
        const CuttingPolarity &polarity,
        const ClassificationCounts &counts,
        int numDimensions)
    {
        GroupCentroids centroids(numDimensions);
        const size_t np = projections.size();

        // Acumuladores
        Eigen::VectorXd sumCorrectLow = Eigen::VectorXd::Zero(numDimensions);
        Eigen::VectorXd sumErrorsLow = Eigen::VectorXd::Zero(numDimensions);
        Eigen::VectorXd sumCorrectHigh = Eigen::VectorXd::Zero(numDimensions);
        Eigen::VectorXd sumErrorsHigh = Eigen::VectorXd::Zero(numDimensions);

        for (size_t i = 0; i < np; ++i)
        {
            int vote = votes[i];
            if (vote == VoteCode::MISSING)
                continue;

            double proj = projections[i];
            int origIdx = originalIndices[i];

            // Extraer coordenadas del legislador
            Eigen::VectorXd coords(numDimensions);
            for (int k = 0; k < numDimensions; ++k)
            {
                coords(k) = legislatorCoords(origIdx, k);
            }

            // Clasificar segun posicion respecto al corte
            if (proj < cutPosition)
            {
                // Lado bajo
                if (vote == polarity.lowSideVote)
                {
                    // Correcto en lado bajo: XJCL
                    sumCorrectLow += coords;
                }
                else if (vote == polarity.highSideVote)
                {
                    // Error en lado bajo: XJEL
                    sumErrorsLow += coords;
                }
            }
            else if (proj > cutPosition)
            {
                // Lado alto
                if (vote == polarity.highSideVote)
                {
                    // Correcto en lado alto: XJCH
                    sumCorrectHigh += coords;
                }
                else if (vote == polarity.lowSideVote)
                {
                    // Error en lado alto: XJEH
                    sumErrorsHigh += coords;
                }
            }
        }

        // Calcular promedios
        if (counts.correctLow > 0)
        {
            centroids.correctLow = sumCorrectLow / counts.correctLow;
        }
        if (counts.errorsLow > 0)
        {
            centroids.errorsLow = sumErrorsLow / counts.errorsLow;
        }
        if (counts.correctHigh > 0)
        {
            centroids.correctHigh = sumCorrectHigh / counts.correctHigh;
        }
        if (counts.errorsHigh > 0)
        {
            centroids.errorsHigh = sumErrorsHigh / counts.errorsHigh;
        }

        return centroids;
    }

    /**
     * Calcula errores de clasificacion por legislador.
     *
     * @param projections Proyecciones ordenadas
     * @param votes Votos reordenados
     * @param originalIndices Mapeo a indices originales
     * @param cutPosition Posicion del corte
     * @param polarity Polaridad del corte
     * @param numLegislators Numero total de legisladores
     * @return Vector de errores: 0=correcto, 1=error, -1=ausente
     */
    std::vector<int> computeLegislatorErrors(
        const std::vector<double> &projections,
        const std::vector<int> &votes,
        const std::vector<int> &originalIndices,
        double cutPosition,
        const CuttingPolarity &polarity,
        int numLegislators)
    {
        std::vector<int> errors(numLegislators, -1); // -1 = no evaluado/ausente
        const size_t np = projections.size();

        for (size_t i = 0; i < np; ++i)
        {
            int vote = votes[i];
            if (vote == VoteCode::MISSING)
                continue;

            double proj = projections[i];
            int origIdx = originalIndices[i];

            if (proj < cutPosition)
            {
                // Lado bajo
                if (vote == polarity.lowSideVote)
                {
                    errors[origIdx] = 0; // Correcto
                }
                else if (vote == polarity.highSideVote)
                {
                    errors[origIdx] = 1; // Error
                }
            }
            else if (proj > cutPosition)
            {
                // Lado alto
                if (vote == polarity.highSideVote)
                {
                    errors[origIdx] = 0; // Correcto
                }
                else if (vote == polarity.lowSideVote)
                {
                    errors[origIdx] = 1; // Error
                }
            }
        }

        return errors;
    }

} // namespace anonimo


// Implementacion de funciones publicas

CuttingPointResult findCuttingPoint1D(
    const std::vector<double> &projections,
    const std::vector<int> &votes,
    const std::vector<int> &originalIndices,
    const Eigen::MatrixXd &legislatorCoords,
    int numDimensions,
    int rollCallIndex,
    CuttingPointMode mode)
{
    CuttingPointResult result(numDimensions);
    result.numLegislators = static_cast<int>(projections.size());
    result.numDimensions = numDimensions;

    if (projections.empty() || projections.size() != votes.size())
    {
        return result;
    }

    // Determinar numero de polaridades a probar
    int numPolaritiesToTest = (mode == CuttingPointMode::SINGLE_POLARITY) ? 1 : 2;

    // Flag para almacenar candidatos intermedios
    bool storeIntermediates = (mode == CuttingPointMode::ROTATION_MODE);

    // Resultados por polaridad
    PolarityResult result1, result2;

    // Evaluar primera polaridad: KCUT=1, LCUT=6
    CuttingPointCandidates *candidatesPtr =
        storeIntermediates ? &result.candidates : nullptr;

    result1 = evaluatePolarityCuts(
        projections, votes,
        VoteCode::YES, VoteCode::NO,
        candidatesPtr);

    // Evaluar segunda polaridad si es necesario: KCUT=6, LCUT=1
    if (numPolaritiesToTest == 2)
    {
        result2 = evaluatePolarityCuts(
            projections, votes,
            VoteCode::NO, VoteCode::YES,
            nullptr); // Solo almacenar candidatos de primera polaridad
    }
    else
    {
        result2.errorRate = 999.0; // Invalidar segunda polaridad
    }

    // Seleccionar polaridad optima
    PolarityResult *best = nullptr;
    if (mode == CuttingPointMode::SINGLE_POLARITY)
    {
        // Forzar primera polaridad
        best = &result1;
        result.polarity = CuttingPolarity(VoteCode::YES, VoteCode::NO);
    }
    else if (result1.errorRate <= result2.errorRate)
    {
        best = &result1;
        result.polarity = CuttingPolarity(VoteCode::YES, VoteCode::NO);
    }
    else
    {
        best = &result2;
        result.polarity = CuttingPolarity(VoteCode::NO, VoteCode::YES);
    }

    // Almacenar resultados finales
    result.cuttingPoint = best->cutPosition;
    result.errorRate = best->errorRate;
    result.counts.correctLow = best->correctLow;
    result.counts.errorsLow = best->errorsLow;
    result.counts.correctHigh = best->correctHigh;
    result.counts.errorsHigh = best->errorsHigh;

    // Calcular centroides si modo NORMAL
    if (mode == CuttingPointMode::NORMAL)
    {
        result.centroids = computeGroupCentroids(
            projections, votes, originalIndices, legislatorCoords,
            result.cuttingPoint, result.polarity, result.counts,
            numDimensions);
    }

    // Calcular errores por legislador si modo NORMAL o SINGLE_POLARITY
    if (mode == CuttingPointMode::NORMAL ||
        mode == CuttingPointMode::SINGLE_POLARITY)
    {
        int maxLegislator = 0;
        for (int idx : originalIndices)
        {
            if (idx > maxLegislator)
                maxLegislator = idx;
        }
        result.legislatorErrors = computeLegislatorErrors(
            projections, votes, originalIndices,
            result.cuttingPoint, result.polarity,
            maxLegislator + 1);
    }

    return result;
}

CuttingPointResult findCuttingPoint1DSimple(
    const std::vector<double> &projections,
    const std::vector<int> &votes)
{
    CuttingPointResult result;
    result.numLegislators = static_cast<int>(projections.size());
    result.numDimensions = 1;

    if (projections.empty() || projections.size() != votes.size())
    {
        return result;
    }

    // Evaluar ambas polaridades
    PolarityResult result1 = evaluatePolarityCuts(
        projections, votes,
        VoteCode::YES, VoteCode::NO,
        nullptr);

    PolarityResult result2 = evaluatePolarityCuts(
        projections, votes,
        VoteCode::NO, VoteCode::YES,
        nullptr);

    // Seleccionar mejor polaridad
    PolarityResult *best = nullptr;
    if (result1.errorRate <= result2.errorRate)
    {
        best = &result1;
        result.polarity = CuttingPolarity(VoteCode::YES, VoteCode::NO);
    }
    else
    {
        best = &result2;
        result.polarity = CuttingPolarity(VoteCode::NO, VoteCode::YES);
    }

    // Almacenar resultados
    result.cuttingPoint = best->cutPosition;
    result.errorRate = best->errorRate;
    result.counts.correctLow = best->correctLow;
    result.counts.errorsLow = best->errorsLow;
    result.counts.correctHigh = best->correctHigh;
    result.counts.errorsHigh = best->errorsHigh;

    return result;
}

/**
 * Encuentra el punto de corte optimo con polaridad fija (JAN11PT).
 *
 * @param projections Proyecciones ordenadas (YSS en Fortran)
 * @param votes Votos ordenados (KA en Fortran)
 * @param polarity Polaridad fija (KCCUT/LCCUT en Fortran)
 * @return Resultado con punto de corte optimo
 */
CuttingPointResult findCuttingPoint1DFixedPolarity(
    const std::vector<double> &projections,
    const std::vector<int> &votes,
    const CuttingPolarity &polarity)
{
    // Inicializacion
    CuttingPointResult result;
    result.numLegislators = static_cast<int>(projections.size());
    result.numDimensions = 1;
    result.polarity = polarity; // Usar polaridad fija pasada como parametro

    if (projections.empty() || projections.size() != votes.size())
    {
        return result;
    }

    // Evaluar solo la polaridad indicada (NOTE=1)
    PolarityResult polarityResult = evaluatePolarityCuts(
        projections,
        votes,
        polarity.lowSideVote,  // KCUT
        polarity.highSideVote, // LCUT
        nullptr);              // No almacenar candidatos intermedios

    // Asignacion de resultados
    result.cuttingPoint = polarityResult.cutPosition;       // AB -> WS(IVOT)
    result.errorRate = polarityResult.errorRate;            // AA
    result.counts.correctLow = polarityResult.correctLow;   // LA -> JCL
    result.counts.errorsLow = polarityResult.errorsLow;     // LB -> JEL
    result.counts.correctHigh = polarityResult.correctHigh; // LC -> JCH
    result.counts.errorsHigh = polarityResult.errorsHigh;   // LD -> JEH

    return result;
}

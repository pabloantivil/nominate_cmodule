/**
 * @file cutting_plane.cpp
 * @brief Implementacion de algoritmos de plano de corte (SEARCH).
 *
 * Subrutinas de plano de corte del codigo Fortran.
 *:
 * - SEARCH -> refineCuttingPlane()
 */

#include "cutting_plane.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>

namespace
{

    /**
     * @brief Proyecta legisladores sobre el vector normal.
     *
     * Calcula: projections[i] = sum_k(legislatorCoords[i,k] * normalVector[k])
     *
     * @param legislatorCoords Coordenadas de legisladores (NP x NS)
     * @param normalVector Vector normal (NS)
     * @return Vector de proyecciones (NP)
     */
    std::vector<double> computeProjections(
        const Eigen::MatrixXd &legislatorCoords,
        const Eigen::VectorXd &normalVector)
    {
        const int np = static_cast<int>(legislatorCoords.rows());
        std::vector<double> projections(np);

        for (int i = 0; i < np; ++i)
        {
            projections[i] = legislatorCoords.row(i).dot(normalVector);
        }

        return projections;
    }

    /**
     * @brief Prepara votos para clasificacion (codigo 0 -> 9).
     *
     * @param votes Vector de votos originales
     * @return Vector de votos con 0 convertido a 9
     */
    std::vector<int> prepareVotes(const std::vector<int> &votes)
    {
        std::vector<int> prepared(votes.size());
        for (size_t i = 0; i < votes.size(); ++i)
        {
            prepared[i] = (votes[i] == 0) ? VoteCode::MISSING : votes[i];
        }
        return prepared;
    }

    /**
     * @brief Reordena votos segun indices de ordenamiento.
     *
     * @param votes Votos originales
     * @param sortedIndices Indices de ordenamiento
     * @return Votos reordenados
     */
    std::vector<int> reorderVotes(
        const std::vector<int> &votes,
        const std::vector<size_t> &sortedIndices)
    {
        std::vector<int> reordered(votes.size());
        for (size_t i = 0; i < sortedIndices.size(); ++i)
        {
            reordered[i] = votes[sortedIndices[i]];
        }
        return reordered;
    }

    /**
     * @brief Reordena proyecciones segun indices de ordenamiento.
     *
     * @param projections Proyecciones originales
     * @param sortedIndices Indices de ordenamiento
     * @return Proyecciones ordenadas
     */
    std::vector<double> reorderProjections(
        const std::vector<double> &projections,
        const std::vector<size_t> &sortedIndices)
    {
        std::vector<double> ordered(projections.size());
        for (size_t i = 0; i < sortedIndices.size(); ++i)
        {
            ordered[i] = projections[sortedIndices[i]];
        }
        return ordered;
    }

    /**
     * @brief Calcula tamano de nube parcial.
     *
     * Formula: KASTRO = max(4*NS, min(4*errores, NP))
     *
     * @param totalErrors Numero total de errores
     * @param numDimensions Numero de dimensiones (NS)
     * @param numLegislators Numero de legisladores (NP)
     * @return Tamano de la nube parcial
     */
    int computePartialCloudSize(int totalErrors, int numDimensions, int numLegislators)
    {
        int kastro = 4 * totalErrors;
        if (kastro > numLegislators)
            kastro = numLegislators;
        if (kastro < 4 * numDimensions)
            kastro = 4 * numDimensions;
        return kastro;
    }

    /**
     * @brief Construye nube completa de puntos (Y16MIDP).
     *
     * @param legislatorCoords Coordenadas de legisladores (NP x NS)
     * @param projections Proyecciones actuales
     * @param votes Votos (codigos: 1=Si, 6=No, 0/9=Ausente)
     * @param normalVector Vector normal actual
     * @param cuttingPoint Punto de corte actual
     * @param polarity Polaridad del corte (KCUT, LCUT)
     * @param wrongFlags Vector de salida indicando legisladores incorrectos (LWRONG)
     * @return Matriz de nube completa (NP x NS)
     */
    Eigen::MatrixXd buildFullPointCloud(
        const Eigen::MatrixXd &legislatorCoords,
        const std::vector<double> &projections,
        const std::vector<int> &votes,
        const Eigen::VectorXd &normalVector,
        double cuttingPoint,
        const CuttingPolarity &polarity,
        std::vector<int> &wrongFlags)
    {
        const int np = static_cast<int>(legislatorCoords.rows());
        const int ns = static_cast<int>(legislatorCoords.cols());

        Eigen::MatrixXd cloud(np, ns);
        wrongFlags.resize(np, 0);

        int kcut = polarity.lowSideVote;
        int lcut = polarity.highSideVote;

        for (int i = 0; i < np; ++i)
        {
            // Distancia al plano: DB2B1 = WS(JX) - XXY(I)
            double db2b1 = cuttingPoint - projections[i];
            int vote = votes[i];
            // Tratar 0 como ausente (9)
            if (vote == 0)
                vote = VoteCode::MISSING;

            bool isLowSide = projections[i] < cuttingPoint;

            if (isLowSide)
            {
                // Lado bajo (proyeccion < corte)
                if (vote == kcut)
                {
                    // Correcto: proyectar al plano
                    // Fortran: Y16MIDP(I,K)=XMAT(I,K)+DB2B1*ZVEC(JX,K)
                    for (int k = 0; k < ns; ++k)
                    {
                        cloud(i, k) = legislatorCoords(i, k) + db2b1 * normalVector(k);
                    }
                }
                else if (vote == lcut)
                {
                    // Incorrecto: mantener posicion original
                    wrongFlags[i] = 1;
                    for (int k = 0; k < ns; ++k)
                    {
                        cloud(i, k) = legislatorCoords(i, k);
                    }
                }
                else
                {
                    // Ausente: proyectar al plano
                    for (int k = 0; k < ns; ++k)
                    {
                        cloud(i, k) = legislatorCoords(i, k) + db2b1 * normalVector(k);
                    }
                }
            }
            else
            {
                // Lado alto (proyeccion > corte)
                if (vote == lcut)
                {
                    // Correcto: proyectar al plano
                    for (int k = 0; k < ns; ++k)
                    {
                        cloud(i, k) = legislatorCoords(i, k) + db2b1 * normalVector(k);
                    }
                }
                else if (vote == kcut)
                {
                    // Incorrecto: mantener posicion original
                    wrongFlags[i] = 1;
                    for (int k = 0; k < ns; ++k)
                    {
                        cloud(i, k) = legislatorCoords(i, k);
                    }
                }
                else
                {
                    // Ausente: proyectar al plano
                    for (int k = 0; k < ns; ++k)
                    {
                        cloud(i, k) = legislatorCoords(i, k) + db2b1 * normalVector(k);
                    }
                }
            }
        }

        return cloud;
    }

    /**
     * @brief Centra una nube de puntos (resta la media de cada dimension).
     *
     * @param cloud Matriz de puntos a centrar (modifica in-place)
     */
    void centerPointCloud(Eigen::MatrixXd &cloud)
    {
        const int np = static_cast<int>(cloud.rows());
        const int ns = static_cast<int>(cloud.cols());

        for (int k = 0; k < ns; ++k)
        {
            double sum = 0.0;
            for (int i = 0; i < np; ++i)
            {
                sum += cloud(i, k);
            }
            double mean = sum / static_cast<double>(np);
            for (int i = 0; i < np; ++i)
            {
                cloud(i, k) -= mean;
            }
        }
    }

    /**
     * @brief Construye nube parcial de puntos (X16MIDP).
     *
     * @param fullCloud Nube completa (Y16MIDP)
     * @param wrongFlags Marcas de legisladores incorrectos
     * @param kastro Numero de puntos a incluir
     * @return Nube parcial (kastro x NS)
     */
    Eigen::MatrixXd buildPartialPointCloud(
        const Eigen::MatrixXd &fullCloud,
        const std::vector<int> &wrongFlags,
        int kastro)
    {
        const int np = static_cast<int>(fullCloud.rows());
        const int ns = static_cast<int>(fullCloud.cols());

        Eigen::MatrixXd partial(kastro, ns);
        int kk = 0;

        // Primero: agregar todos los incorrectos
        for (int i = 0; i < np && kk < kastro; ++i)
        {
            if (wrongFlags[i] == 1)
            {
                partial.row(kk) = fullCloud.row(i);
                ++kk;
            }
        }

        // Despues: agregar correctos hasta completar kastro
        for (int i = 0; i < np && kk < kastro; ++i)
        {
            if (wrongFlags[i] == 0)
            {
                partial.row(kk) = fullCloud.row(i);
                ++kk;
            }
        }

        return partial;
    }

    /**
     * @brief Calcula la direccion de minima varianza usando SVD.
     *
     * La SVD descompone la matriz: A = U * S * V^T
     *
     * El ultimo vector singular derecho (ultima columna de V) corresponde
     * a la direccion de minima varianza, que es el nuevo vector normal.
     * 
     * @param cloud Matriz de puntos centrada (M x NS)
     * @return Vector de direccion de minima varianza (NS)
     */
    Eigen::VectorXd computeMinVarianceDirection(const Eigen::MatrixXd &cloud)
    {
        const int ns = static_cast<int>(cloud.cols());

        // SVD de la nube de puntos
        // Se usa JacobiSVD que es equivalente a DGESDD
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(cloud, Eigen::ComputeThinV);

        // El ultimo vector singular derecho (columna NS-1 de V)
        // corresponde a la direccion de minima varianza
        Eigen::VectorXd minVarianceDir = svd.matrixV().col(ns - 1);

        return minVarianceDir;
    }

    /**
     * @brief Selecciona la mejor iteracion basado en numero de errores.
     *
     * Algoritmo:
     * 1. Ordena iteraciones por numero de errores
     * 2. Encuentra el indice con minimo errores
     * 3. Si hay empates, el ordenamiento determina cual se selecciona
     *
     * @param states Vector de estados de iteracion
     * @return Indice de la mejor iteracion
     */
    size_t selectBestIteration(const std::vector<SearchIterationState> &states)
    {
        if (states.empty())
            return 0;

        const size_t n = states.size();

        // Crear vector de errores para ordenar
        std::vector<double> errorCounts(n);
        for (size_t i = 0; i < n; ++i)
        {
            errorCounts[i] = states[i].errorCount;
        }

        // Ordenar indices por errores (ascendente)
        std::vector<size_t> sortedIndices = argsort(errorCounts);

        // El primer indice tiene el minimo de errores
        // Buscar empates y tomar el ultimo con el mismo valor minimo
        size_t bestIdx = sortedIndices[0];
        double minErrors = states[bestIdx].errorCount;

        // Buscar el ultimo con el mismo numero de errores
        size_t lastSame = 0;
        for (size_t j = 0; j < n; ++j)
        {
            if (std::abs(errorCounts[sortedIndices[j]] - minErrors) < 0.0001)
            {
                lastSame = j;
            }
            else
            {
                break;
            }
        }

        // Fortran: KIN = JJ - 1, luego LLM(1) = LLM(KIN)
        // Toma el ultimo elemento empatado antes del primero diferente
        bestIdx = sortedIndices[lastSame];

        return bestIdx;
    }

} // namespace anonimo

// ============================================================================
// Implementacion de funciones publicas
// ============================================================================

SearchResult refineCuttingPlane(
    const Eigen::MatrixXd &legislatorCoords,
    const std::vector<int> &votes,
    const Eigen::VectorXd &initialNormal,
    double initialCutPoint,
    const CuttingPolarity &initialPolarity,
    int maxIterations)
{
    const int np = static_cast<int>(legislatorCoords.rows());
    const int ns = static_cast<int>(legislatorCoords.cols());

    // Validacion de entrada
    if (np == 0 || ns == 0 || static_cast<int>(votes.size()) != np)
    {
        SearchResult emptyResult(ns);
        emptyResult.normalVector = initialNormal;
        emptyResult.cuttingPoint = initialCutPoint;
        emptyResult.polarity = initialPolarity;
        return emptyResult;
    }

    // Inicializar estado
    Eigen::VectorXd currentNormal = initialNormal;
    CuttingPolarity currentPolarity = initialPolarity;

    // Preparar votos (0 -> 9)
    std::vector<int> preparedVotes = prepareVotes(votes);

    // Vector para almacenar estados de cada iteracion
    // UUU, FV1, FV2, KKKCUT, LLLCUT, LLN
    std::vector<SearchIterationState> iterationStates;
    iterationStates.reserve(maxIterations);

    // Variables para clasificacion
    int lastJCH = 0, lastJEH = 0, lastJCL = 0, lastJEL = 0;

    // Loop principal de busqueda 
    for (int iter = 0; iter < maxIterations; ++iter)
    {
        SearchIterationState state;
        state.normalVector = currentNormal;

        // Proyeccion y clasificacion
        // Proyectar legisladores sobre el vector normal actual
        std::vector<double> projections = computeProjections(legislatorCoords, currentNormal);

        // Ordenar proyecciones
        std::vector<size_t> sortedIndices = argsort(projections);
        std::vector<double> sortedProjections = reorderProjections(projections, sortedIndices);
        std::vector<int> sortedVotes = reorderVotes(preparedVotes, sortedIndices);

        // Convertir indices a int para findCuttingPoint1D
        std::vector<int> originalIndices(np);
        for (int i = 0; i < np; ++i)
        {
            originalIndices[i] = static_cast<int>(sortedIndices[i]);
        }

        // Encontrar punto de corte optimo usando JAN1PT
        CuttingPointResult cutResult = findCuttingPoint1D(
            sortedProjections,
            sortedVotes,
            originalIndices,
            legislatorCoords,
            ns,
            0, 
            CuttingPointMode::NORMAL);

        // Guardar resultados de iteracion
        state.errorCount = static_cast<double>(cutResult.counts.totalErrors());
        state.cuttingPoint = cutResult.cuttingPoint;
        state.kcut = cutResult.polarity.lowSideVote;
        state.lcut = cutResult.polarity.highSideVote;
        state.totalErrors = cutResult.counts.totalErrors();
        state.counts = cutResult.counts;

        // Actualizar polaridad actual
        currentPolarity = cutResult.polarity;

        // Guardar conteos para uso posterior
        lastJCH = cutResult.counts.correctHigh;
        lastJEH = cutResult.counts.errorsHigh;
        lastJCL = cutResult.counts.correctLow;
        lastJEL = cutResult.counts.errorsLow;

        iterationStates.push_back(state);

        // Retorno temprano si clasificacion perfecta
        if (cutResult.counts.totalErrors() == 0)
        {
            SearchResult result(ns);
            result.normalVector = currentNormal;
            result.cuttingPoint = cutResult.cuttingPoint;
            result.polarity = currentPolarity;
            result.counts = cutResult.counts;
            result.errors = 0;
            result.totalClassified = cutResult.counts.total();
            result.projections = projections;
            result.bestIteration = iter;
            result.isPerfectClassification = true;
            return result;
        }

        // Determinar tamano de nube parcial
        int totalErrors = cutResult.counts.totalErrors();
        int kastro = computePartialCloudSize(totalErrors, ns, np);

        // Construir nube completa Y16MIDP
        std::vector<int> wrongFlags;
        Eigen::MatrixXd fullCloud = buildFullPointCloud(
            legislatorCoords,
            projections,
            preparedVotes,
            currentNormal,
            cutResult.cuttingPoint,
            currentPolarity,
            wrongFlags);

        // Centrar nube completa
        centerPointCloud(fullCloud);

        // Construir nube parcial X16MIDP
        Eigen::MatrixXd partialCloud = buildPartialPointCloud(fullCloud, wrongFlags, kastro);

        // Centrar nube parcial
        centerPointCloud(partialCloud);

        // SVD sobre nube completa
        // Actualizar vector normal con direccion de minima varianza de nube completa
        currentNormal = computeMinVarianceDirection(fullCloud);

        // SVD sobre nube parcial (lineas 3335-3343)
        // Si iter > 25, usar la nube parcial en su lugar
        if (iter > 25)
        {
            currentNormal = computeMinVarianceDirection(partialCloud);
        }
    }

    // Seleccionar mejor iteracion
    size_t bestIterIdx = selectBestIteration(iterationStates);

    const SearchIterationState &bestState = iterationStates[bestIterIdx];

    // Recalcular proyecciones finales
    std::vector<double> finalProjections = computeProjections(
        legislatorCoords, bestState.normalVector);

    // Construir resultado final
    SearchResult result(ns);
    result.normalVector = bestState.normalVector;
    result.cuttingPoint = bestState.cuttingPoint;
    result.polarity = CuttingPolarity(bestState.kcut, bestState.lcut);
    result.counts = bestState.counts;
    result.errors = bestState.totalErrors;
    result.totalClassified = bestState.counts.total();
    result.projections = finalProjections;
    result.bestIteration = static_cast<int>(bestIterIdx);
    result.isPerfectClassification = false;

    return result;
}

bool refineCuttingPlaneInPlace(
    const Eigen::MatrixXd &legislatorCoords,
    const std::vector<int> &votes,
    Eigen::VectorXd &normalVector,
    double &cuttingPoint,
    CuttingPolarity &polarity,
    std::vector<double> &projections,
    int maxIterations,
    int &outErrors,
    int &outTotalClassified)
{
    SearchResult result = refineCuttingPlane(
        legislatorCoords,
        votes,
        normalVector,
        cuttingPoint,
        polarity,
        maxIterations);

    // Actualizar parametros de salida
    normalVector = result.normalVector;
    cuttingPoint = result.cuttingPoint;
    polarity = result.polarity;
    projections = result.projections;
    outErrors = result.errors;
    outTotalClassified = result.totalClassified;

    return result.isPerfectClassification;
}

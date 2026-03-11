/**
 * @file cutting_plane.cpp
 * Implementacion de algoritmos de plano de corte (SEARCH/CUTPLANE).
 */

#include "cutting_plane.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>

namespace
{
    /**
     * Proyecta legisladores sobre el vector normal.
     * Calcula: projections[i] = sum_k(legislatorCoords[i,k] * normalVector[k])
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

    // Prepara votos para clasificacion (codigo 0 -> 9)
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
     * Reordena votos segun indices de ordenamiento.
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
     * Reordena proyecciones segun indices de ordenamiento.
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
     * Calcula tamano de nube parcial.
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
     * Construye nube completa de puntos (Y16MIDP).
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
     * Centra una nube de puntos (resta la media de cada dimension).
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
     * Construye nube parcial de puntos (X16MIDP).
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
     * Calcula la direccion de minima varianza usando SVD.
     * @param cloud Matriz de puntos centrada (M x NS)
     * @return Vector de direccion de minima varianza (NS)
     */
    Eigen::VectorXd computeMinVarianceDirection(const Eigen::MatrixXd &cloud)
    {
        const int ns = static_cast<int>(cloud.cols());

        // SVD de la nube de puntos
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(cloud, Eigen::ComputeThinV);

        // El ultimo vector singular derecho (columna NS-1 de V)
        // corresponde a la direccion de minima varianza
        Eigen::VectorXd minVarianceDir = svd.matrixV().col(ns - 1);

        return minVarianceDir;
    }

    /**
     * Selecciona la mejor iteracion basado en numero de errores.
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

        bestIdx = sortedIndices[lastSame];

        return bestIdx;
    }

} // namespace anonimo

// Implementacion de funciones publicas

// Flag global para debug de SEARCH (solo para tests)
bool g_searchDebug = false;

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

        // SVD sobre nube parcial
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

// CUTPLANE - Orquestador de clasificacion inicial de todas las votaciones
namespace
{
    /**
     * Proyecta un legislador sobre el vector normal.
     * @param legislatorCoords Fila de coordenadas del legislador
     * @param normalVector Vector normal
     * @return Valor de la proyeccion escalar
     */
    inline double computeSingleProjection(
        const Eigen::RowVectorXd &legislatorCoords,
        const Eigen::VectorXd &normalVector)
    {
        return legislatorCoords.dot(normalVector);
    }

    /**
     * Clasifica errores para una votacion y calcula estadisticas.
     * @param projections Proyecciones finales de legisladores (XXY)
     * @param votes Votos originales (LDATA(:,JX))
     * @param cuttingPoint Punto de corte (WS(JX))
     * @param polarity Polaridad (KCUT, LCUT)
     * @param outLegislatorErrors Vector de errores por legislador (salida)
     * @param outStats Estadisticas de proyecciones (salida)
     * @return Numero total de errores (KSUM)
     */
    int computeClassificationErrors(
        const std::vector<double> &projections,
        const std::vector<int> &votes,
        double cuttingPoint,
        const CuttingPolarity &polarity,
        std::vector<int> &outLegislatorErrors,
        ProjectionStats &outStats)
    {
        const int np = static_cast<int>(projections.size());
        outLegislatorErrors.assign(np, 0);

        int ksum = 0;        // Total de errores
        double sumYes = 0.0; // Suma de proyecciones de votantes "Si"
        double sumNo = 0.0;  // Suma de proyecciones de votantes "No"
        int ksumYes = 0;     // Contador de votantes "Si"
        int ksumNo = 0;      // Contador de votantes "No"

        int kcut = polarity.lowSideVote;
        int lcut = polarity.highSideVote;

        for (int i = 0; i < np; ++i)
        {
            int vote = votes[i];

            // Ausentes no generan errores
            if (vote == 0 || vote == VoteCode::MISSING)
            {
                continue;
            }

            // Acumular estadisticas de proyecciones
            if (vote == VoteCode::YES)
            {
                sumYes += projections[i];
                ksumYes++;
            }
            else if (vote == VoteCode::NO)
            {
                sumNo += projections[i];
                ksumNo++;
            }

            // Determinar si hay error de clasificacion
            if (projections[i] < cuttingPoint)
            {
                if (vote != kcut)
                {
                    outLegislatorErrors[i] = 1;
                    ksum++;
                }
            }
            else if (projections[i] > cuttingPoint)
            {
                if (vote != lcut)
                {
                    outLegislatorErrors[i] = 1;
                    ksum++;
                }
            }
            // Si proyeccion == cuttingPoint exactamente, no hay error
            // (comportamiento implicito del Fortran)
        }

        // Calcular medias
        outStats.meanYes = (ksumYes > 0) ? sumYes / ksumYes : 0.0;
        outStats.meanNo = (ksumNo > 0) ? sumNo / ksumNo : 0.0;
        outStats.countYes = ksumYes;
        outStats.countNo = ksumNo;

        return ksum;
    }

} // namespace anonimo (CUTPLANE helpers)

// classifyRollCall - Clasificacion de una votacion individual
RollCallClassification classifyRollCall(
    const Eigen::MatrixXd &legislatorCoords,
    Eigen::VectorXd &normalVector,
    const std::vector<int> &votes,
    bool searchEnabled)
{
    const int np = static_cast<int>(legislatorCoords.rows());
    const int ns = static_cast<int>(legislatorCoords.cols());

    RollCallClassification result;
    result.legislatorErrors.resize(np, 0);
    result.projections.resize(np);
    result.searchPerformed = false;

    // Contar votos Si y No
    int kyes = 0;
    int kno = 0;
    for (int i = 0; i < np; ++i)
    {
        if (votes[i] == VoteCode::YES)
            kyes++;
        else if (votes[i] == VoteCode::NO)
            kno++;
    }

    // Calcular proyecciones sobre vector normal
    std::vector<double> projections(np);
    std::vector<int> preparedVotes(np);

    for (int i = 0; i < np; ++i)
    {
        projections[i] = computeSingleProjection(
            legislatorCoords.row(i), normalVector);

        preparedVotes[i] = (votes[i] == 0) ? VoteCode::MISSING : votes[i];
    }

    // Ordenar proyecciones
    std::vector<size_t> sortedIndices = argsort(projections);
    std::vector<double> sortedProjections(np);
    std::vector<int> sortedVotes(np);
    std::vector<int> originalIndices(np);

    for (int i = 0; i < np; ++i)
    {
        sortedProjections[i] = projections[sortedIndices[i]];
        sortedVotes[i] = preparedVotes[sortedIndices[i]];
        originalIndices[i] = static_cast<int>(sortedIndices[i]);
    }

    // Encontrar punto de corte optimo - JAN1PT
    CuttingPointResult cutResult = findCuttingPoint1D(
        sortedProjections,
        sortedVotes,
        originalIndices,
        legislatorCoords,
        ns,
        0, // rollCallIndex (no usado en modo NORMAL)
        CuttingPointMode::NORMAL);

    // Extraer resultados de JAN1PT
    result.polarity = cutResult.polarity;
    result.cuttingPoint = cutResult.cuttingPoint;
    result.counts = cutResult.counts;

    int jeh = cutResult.counts.errorsHigh;
    int jel = cutResult.counts.errorsLow;
    int jch = cutResult.counts.correctHigh;
    int jcl = cutResult.counts.correctLow;

    // Decision de busqueda adicional
    if (jeh + jel == 0)
    {
        // Clasificacion perfecta: no se necesita SEARCH
        result.totalClassified = jch + jeh + jcl + jel;
        result.totalErrors = 0;
        result.searchPerformed = false;
    }
    else if (!searchEnabled)
    {
        // Sin busqueda: usar resultado de JAN1PT tal cual
        result.totalClassified = jch + jeh + jcl + jel;
        result.totalErrors = jeh + jel;
        result.searchPerformed = false;
    }
    // Busqueda de rotaciones - SEARCH
    else
    {
        // Solo ejecutar SEARCH si NS > 1
        if (ns > 1)
        {
            SearchResult searchResult = refineCuttingPlane(
                legislatorCoords,
                votes,
                normalVector,
                cutResult.cuttingPoint,
                cutResult.polarity,
                25); // NCUT=25

            // Actualizar vector normal (in-place, similar al Fortran)
            normalVector = searchResult.normalVector;

            // Usar resultados de SEARCH
            result.polarity = searchResult.polarity;
            result.cuttingPoint = searchResult.cuttingPoint;
            result.counts = searchResult.counts;
            result.totalErrors = searchResult.errors;
            result.totalClassified = searchResult.totalClassified;
            result.searchPerformed = true;

            // Actualizar proyecciones con el nuevo vector normal
            for (int i = 0; i < np; ++i)
            {
                projections[i] = computeSingleProjection(
                    legislatorCoords.row(i), normalVector);
            }
        }
        else
        {
            // NS=1: no hay SEARCH posible
            result.totalClassified = jch + jeh + jcl + jel;
            result.totalErrors = jeh + jel;
            result.searchPerformed = false;
        }
    }

    // Almacenar proyecciones finales
    result.projections = projections;

    // Calcular y almacenar errores de clasificacion
    int errorCount = computeClassificationErrors(
        result.projections,
        votes,
        result.cuttingPoint,
        result.polarity,
        result.legislatorErrors,
        result.projStats);

    // Actualizar conteo de errores
    result.totalErrors = errorCount;

    return result;
}

// findAllCuttingPlanes: Clasificacion inicial de todas las votaciones (CUTPLANE)
CutplaneResult findAllCuttingPlanes(
    const Eigen::MatrixXd &legislatorCoords,
    Eigen::MatrixXd &normalVectors,
    const Eigen::MatrixXi &voteMatrix,
    bool searchEnabled)
{
    const int np = static_cast<int>(legislatorCoords.rows());
    const int nrcall = static_cast<int>(voteMatrix.cols());
    const int ns = static_cast<int>(legislatorCoords.cols());

    // Inicializacion
    CutplaneResult result;
    result.numLegislators = np;
    result.numRollCalls = nrcall;
    result.numDimensions = ns;
    result.totalClassified = 0;
    result.totalErrors = 0;

    result.polarities.resize(nrcall);
    result.cuttingPoints.resize(nrcall);
    result.rollCallResults.resize(nrcall);

    int ktSave = 0;
    int kttSave = 0;

    // Loop principal sobre todas las votaciones
    for (int jx = 0; jx < nrcall; ++jx)
    {
        // Extraer vector de votos para esta votacion
        std::vector<int> votes(np);
        for (int i = 0; i < np; ++i)
        {
            votes[i] = voteMatrix(i, jx);
        }

        // Extraer vector normal para esta votacion
        Eigen::VectorXd normalVector = normalVectors.row(jx).transpose();

        // Clasificar esta votacion
        RollCallClassification rcResult = classifyRollCall(
            legislatorCoords,
            normalVector,
            votes,
            searchEnabled);

        // Actualizar vector normal en la matriz (si SEARCH lo modifico)
        normalVectors.row(jx) = normalVector.transpose();

        // Almacenar polaridad
        result.polarities[jx] = rcResult.polarity;
        result.cuttingPoints[jx] = rcResult.cuttingPoint;

        // Almacenar resultado completo
        result.rollCallResults[jx] = rcResult;

        ktSave += rcResult.totalClassified;
        kttSave += rcResult.totalErrors;
    }

    // Finalizacion y estadisticas
    result.totalClassified = ktSave;
    result.totalErrors = kttSave;

    if (result.totalClassified > 0)
    {
        result.errorRate = static_cast<double>(result.totalErrors) /
                           static_cast<double>(result.totalClassified);
        result.accuracy = 1.0 - result.errorRate;
    }
    else
    {
        result.errorRate = 0.0;
        result.accuracy = 1.0;
    }

    return result;
}

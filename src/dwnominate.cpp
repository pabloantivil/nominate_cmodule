/**
 * @file dwnominate.cpp
 * @brief Implementacion de la clase DWNominate.
 */

#include "dwnominate.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <limits>

// CONSTRUCTOR E INICIALIZACION
DWNominate::DWNominate(const DWNominateConfig &config, const DWNominateInput &input)
    : config_(config),
      votes_(input.votes),
      currentLogLikelihood_(0.0)
{
    // call init_zdf
    initializeCDF();

    // WEIGHT(1:(NS+1)) = WEIGHTSIN
    int ns = config_.numDimensions;
    weights_ = input.initialWeights;
    if (weights_.size() == 0)
    {
        // Valores por defecto si no se proporcionan
        weights_.resize(ns + 1);
        weights_.setOnes();
        weights_(ns) = 4.925; // Beta por defecto
    }
    // WEIGHT(1)=1.000 (peso dimension 1 siempre es 1)
    weights_(0) = 1.0;

    // Cargar metadata de congresos
    loadCongressMetadata(input);

    // Cargar roll calls
    loadRollCalls(input);

    // Cargar legisladores
    loadLegislators(input);

    // Inicializar matrices de estado
    int numLegislators = static_cast<int>(legislatorCoords_.rows());
    int numRollCalls = static_cast<int>(rollCallMidpoints_.rows());

    // XBIGLOG: Log-likelihood por legislador (antes/despues)
    legislatorLogLikelihood_ = Eigen::MatrixXd::Zero(numLegislators, 2);

    // KBIGLOG: Conteos por legislador
    legislatorVoteCounts_ = Eigen::MatrixXi::Zero(numLegislators, 4);

    // XVAR: Varianzas por legislador unico
    int maxUniqueId = 0;
    for (int id : legislatorUniqueId_)
    {
        maxUniqueId = std::max(maxUniqueId, id);
    }
    legislatorVariances_ = Eigen::MatrixXd::Zero(maxUniqueId + 1, 6);

    // Polaridad de cortes por roll call
    rollCallPolarity_.resize(numRollCalls);

    log("DWNominate inicializado");
    log("  Dimensiones: " + std::to_string(ns));
    log("  Congresos: " + std::to_string(config_.firstCongress) +
        " - " + std::to_string(config_.lastCongress));
    log("  Roll calls: " + std::to_string(numRollCalls));
    log("  Legisladores: " + std::to_string(numLegislators));
}

/**
 * Inicializa la tabla CDF.
 */
void DWNominate::initializeCDF()
{
    // NormalCDF se inicializa en su constructor con los mismos
    // parametros que init_zdf: NDEVIT=50001, XDEVIT=10000.0
    // La tabla ZDF se precomputa automaticamente.
}

/**
 * Carga metadata de congresos y calcula offsets.
 */
void DWNominate::loadCongressMetadata(const DWNominateInput &input)
{
    int numCongresses = config_.lastCongress - config_.firstCongress + 1;
    congressInfo_.resize(numCongresses);

    int legislatorOffset = 0;
    int rollCallOffset = 0;

    for (int i = 0; i < numCongresses; ++i)
    {
        int congressIndex = config_.firstCongress + i;
        CongressInfo &info = congressInfo_[i];

        info.index = congressIndex;

        // Obtener numLegislators y numRollCalls de la metadata
        if (congressIndex < static_cast<int>(input.congressMetadata.size()))
        {
            info.numLegislators = input.congressMetadata[congressIndex].first;
            info.numRollCalls = input.congressMetadata[congressIndex].second;
        }
        else
        {
            info.numLegislators = 0;
            info.numRollCalls = 0;
        }

        // Offsets (equivalen a KTOTP, KTOTQ acumulados)
        info.legislatorOffset = legislatorOffset;
        info.rollCallOffset = rollCallOffset;

        legislatorOffset += info.numLegislators;
        rollCallOffset += info.numRollCalls;
    }
}

/**
 * Carga roll calls y determina validez.
 */
void DWNominate::loadRollCalls(const DWNominateInput &input)
{
    // Copiar roll call congress
    rollCallCongress_ = input.rollCallCongress;

    // Copiar midpoints y spreads
    rollCallMidpoints_ = input.rollCallMidpoints;
    rollCallSpreads_ = input.rollCallSpreads;

    // Determinar validez de cada roll call
    int numRollCalls = static_cast<int>(rollCallCongress_.size());
    validRollCalls_.resize(numRollCalls);

    for (int i = 0; i < numRollCalls; ++i)
    {
        // Contar votos Si y No
        int kyes = 0;
        int kno = 0;

        size_t numLeg = votes_.getNumLegislators();
        for (size_t leg = 0; leg < numLeg; ++leg)
        {
            if (!votes_.isMissing(leg, i))
            {
                if (votes_.getVote(leg, i))
                {
                    kyes++;
                }
                else
                {
                    kno++;
                }
            }
        }

        // Calcular margen
        int krctot = kyes + kno;
        int krcmin = std::min(kyes, kno);
        double xmarg = (krctot > 0) ? static_cast<double>(krcmin) / krctot : 0.0;
        // Nota: RCBAD=.TRUE. significa roll call VALIDO en Fortran
        validRollCalls_[i] = (xmarg >= config_.marginThreshold);
    }
}

/**
 * Carga legisladores y construye mapas de presencia.
 */
void DWNominate::loadLegislators(const DWNominateInput &input)
{
    // Copiar datos basicos
    legislatorCongress_ = input.legislatorCongress;
    legislatorUniqueId_ = input.legislatorUniqueId;
    legislatorCoords_ = input.legislatorCoords;

    // Construir mapa de presencia (reemplaza LWHERE/KWHERE)
    int maxUniqueId = 0;
    for (int id : legislatorUniqueId_)
    {
        maxUniqueId = std::max(maxUniqueId, id);
    }
    legislatorPresence_.resize(maxUniqueId + 1);

    for (int i = 0; i < static_cast<int>(legislatorUniqueId_.size()); ++i)
    {
        int uniqueId = legislatorUniqueId_[i];
        int congress = legislatorCongress_[i];

        // Inicializar si es la primera vez que vemos este ID
        if (legislatorPresence_[uniqueId].uniqueId < 0)
        {
            legislatorPresence_[uniqueId].uniqueId = uniqueId;
        }

        legislatorPresence_[uniqueId].congressToDataIndex[congress] = i;
    }
}

// METODO PRINCIPAL run()
/**
 * Ejecuta el algoritmo completo.
 */
DWNominateResult DWNominate::run()
{
    DWNominateResult result;
    int ns = config_.numDimensions;

    // Bucle principal IHAPPY
    for (int ihappy = config_.firstIteration; ihappy <= config_.lastIteration; ++ihappy)
    {
        log("=== Iteracion global " + std::to_string(ihappy) + " ===");

        // Fase de pesos dimensionales (WINT)
        if (ns >= 2)
        {
            log("Estimando pesos dimensionales...");
            executeWeightPhase();
        }

        // Fase de beta (SIGMAS)
        log("Estimando beta...");
        executeBetaPhase();

        // Fase de roll calls
        log("Estimando vectores de roll calls...");
        executeRollCallPhase(ihappy);

        // PLOG despues de roll calls
        currentLogLikelihood_ = computeLogLikelihood();

        // Fase de legisladores
        log("Estimando coordenadas de legisladores...");
        executeLegislatorPhase();

        // PLOG despues de legisladores
        currentLogLikelihood_ = computeLogLikelihood();

        result.totalIterations = ihappy;
    }
    // Fin bucle 9999

    // Preparar resultados
    result.legislatorCoords = legislatorCoords_;
    result.rollCallMidpoints = rollCallMidpoints_;
    result.rollCallSpreads = rollCallSpreads_;
    result.weights = weights_;
    result.finalLogLikelihood = currentLogLikelihood_;
    result.finalStats = globalStats_;

    // Recopilar estadisticas por legislador
    int numLegislators = static_cast<int>(legislatorCoords_.rows());
    result.legislatorStats.resize(numLegislators);

    for (int i = 0; i < numLegislators; ++i)
    {
        LegislatorStats &stats = result.legislatorStats[i];

        stats.logLikelihoodBefore = legislatorLogLikelihood_(i, 0);
        stats.logLikelihoodAfter = legislatorLogLikelihood_(i, 1);
        stats.voteCountBefore = legislatorVoteCounts_(i, 0);
        stats.voteCountAfter = legislatorVoteCounts_(i, 1);
        stats.yesCountBefore = legislatorVoteCounts_(i, 2);
        stats.yesCountAfter = legislatorVoteCounts_(i, 3);

        // GMP = exp(LL / N)
        if (stats.voteCountBefore > 0)
        {
            stats.gmpBefore = std::exp(stats.logLikelihoodBefore /
                                       stats.voteCountBefore);
        }
        if (stats.voteCountAfter > 0)
        {
            stats.gmpAfter = std::exp(stats.logLikelihoodAfter /
                                      stats.voteCountAfter);
        }

        // Varianzas (si NS=2)
        if (ns == 2)
        {
            int uniqueId = legislatorUniqueId_[i];
            double tt = (legislatorCoords_.cols() > ns)
                            ? legislatorCoords_(i, ns)
                            : 0.0;

            // VARX1=XVAR(ID1(I),1)+TT*TT*XVAR(ID1(I),2)+2.0*TT*XVAR(ID1(I),3)
            stats.varianceX1 = legislatorVariances_(uniqueId, 0) + tt * tt * legislatorVariances_(uniqueId, 1) + 2.0 * tt * legislatorVariances_(uniqueId, 2);

            // VARX2=XVAR(ID1(I),4)+TT*TT*XVAR(ID1(I),5)+2.0*TT*XVAR(ID1(I),6)
            stats.varianceX2 = legislatorVariances_(uniqueId, 3) + tt * tt * legislatorVariances_(uniqueId, 4) + 2.0 * tt * legislatorVariances_(uniqueId, 5);

            stats.stdDevX1 = std::sqrt(std::abs(stats.varianceX1));
            stats.stdDevX2 = std::sqrt(std::abs(stats.varianceX2));
        }
    }

    return result;
}

// FASE DE PESOS (WINT)
/**
 * Esta fase solo se ejecuta si NS >= 2.
 * Optimiza WEIGHT(2:NS) manteniendo WEIGHT(1)=1.0.
 */
void DWNominate::executeWeightPhase()
{
    int ns = config_.numDimensions;

    // Verificacion de seguridad (ya se verifica en run(), pero doble check)
    if (ns < 2)
    {
        return;
    }

    // Construir vector de parametros de roll calls
    std::vector<RollCallParameters> rollCallParams = buildRollCallParams();

    // Crear contexto de likelihood
    // LikelihoodContext toma weights_ por referencia, permitiendo que
    // optimizeWeight2 modifique weights_ directamente
    LikelihoodContext context(
        legislatorCoords_,
        rollCallParams,
        votes_,
        weights_,
        normalCDF_,
        validRollCalls_);

    // Configuracion de WINT (paso = 0.01)
    WeightOptimizerConfig config = wintConfig();
    config.verbose = config_.verbose;

    // Ejecutar optimizacion
    WeightOptimizationResult result = optimizeWeight2(context, config);

    // weights_ ya fue modificado in-place por optimizeWeight2 a traves del context
    // Actualizar log-likelihood actual
    currentLogLikelihood_ = result.logLikelihood;

    if (config_.verbose)
    {
        log("  [WINT] W2: " + std::to_string(result.initialValue) +
            " -> " + std::to_string(result.value) +
            ", LL: " + std::to_string(result.logLikelihood) +
            ", iters: " + std::to_string(result.iterations));
    }
}

// FASE DE BETA (SIGMAS)

/**
 * Ejecuta fase de optimizacion de beta (SIGMAS).
 * Optimiza WEIGHT(NS+1) = beta mediante busqueda grid.
 */
void DWNominate::executeBetaPhase()
{
    // Construir vector de parametros de roll calls
    std::vector<RollCallParameters> rollCallParams = buildRollCallParams();

    // Crear contexto de likelihood
    // LikelihoodContext toma weights_ por referencia, permitiendo que
    // optimizeBeta modifique weights_ directamente
    LikelihoodContext context(
        legislatorCoords_,
        rollCallParams,
        votes_,
        weights_,
        normalCDF_,
        validRollCalls_);

    // Configuracion de SIGMAS (paso = 0.1)
    BetaOptimizerConfig config = sigmasConfig();
    config.verbose = config_.verbose;

    // Ejecutar optimizacion
    BetaOptimizationResult result = optimizeBeta(context, config);

    // weights_ ya fue modificado in-place por optimizeBeta a traves del context
    // Actualizar log-likelihood actual
    currentLogLikelihood_ = result.logLikelihood;

    if (config_.verbose)
    {
        log("  [SIGMAS] Beta: " + std::to_string(result.initialValue) +
            " -> " + std::to_string(result.value) +
            ", LL: " + std::to_string(result.logLikelihood) +
            ", iters: " + std::to_string(result.iterations));
    }
}

// FASE DE ROLL CALLS
void DWNominate::executeRollCallPhase(int iteration)
{
    int classificationBefore = 0; // LASSB4
    int classificationAfter = 0;  // LASSAF
    int totalVotes = 0;           // LATOT

    // Loop sobre congresos
    for (const CongressInfo &congress : congressInfo_)
    {
        int congressIndex = congress.index;

        // Verificar que el congreso esta en el rango
        if (congressIndex < config_.firstCongress ||
            congressIndex > config_.lastCongress)
        {
            continue;
        }

        int legislatorOffset = congress.legislatorOffset;
        int rollCallOffset = congress.rollCallOffset;
        int numRollCalls = congress.numRollCalls;

        // Loop sobre roll calls del congreso
        for (int j = 0; j < numRollCalls; ++j)
        {
            int globalRollCallIndex = rollCallOffset + j;

            // Verificar que el roll call existe
            if (globalRollCallIndex >= static_cast<int>(validRollCalls_.size()))
            {
                continue;
            }

            processRollCall(
                congressIndex,
                j,
                globalRollCallIndex,
                iteration,
                legislatorOffset,
                classificationBefore,
                classificationAfter,
                totalVotes);
        }
        // Fin loop 3 (roll calls)
    }
    // Fin loop 2 (congresos)

    // Calcular estadisticas
    if (totalVotes > 0)
    {
        double yclass = static_cast<double>(classificationAfter) / totalVotes;
        log("  Clasificacion: " + std::to_string(classificationAfter) +
            "/" + std::to_string(totalVotes) +
            " (" + std::to_string(yclass * 100.0) + "%)");
    }
}

/**
 * Procesa un roll call individual.
 */
void DWNominate::processRollCall(
    int congressIndex,
    int rollCallLocalIndex,
    int globalRollCallIndex,
    int iteration,
    int legislatorOffset,
    int &classificationBefore,
    int &classificationAfter,
    int &totalVotes)
{
    int ns = config_.numDimensions;

    // Preparar datos del roll call
    Eigen::MatrixXd coords;
    std::vector<int> voteCodes;
    std::vector<int> sortedIndices;
    int kyes = 0;
    int kno = 0;

    int numVoters = prepareRollCallData(
        congressIndex,
        rollCallLocalIndex,
        legislatorOffset,
        coords,
        voteCodes,
        sortedIndices,
        kyes,
        kno);

    if (numVoters == 0)
    {
        return;
    }

    // Verificar margen
    if (!isRollCallValid(kyes, kno))
    {
        for (int k = 0; k < ns; ++k)
        {
            rollCallMidpoints_(globalRollCallIndex, k) = 0.0;
            rollCallSpreads_(globalRollCallIndex, k) = 0.0;
        }
        return;
    }

    // Normalizar midpoint a esfera unitaria
    Eigen::VectorXd midpoint = rollCallMidpoints_.row(globalRollCallIndex).transpose();
    normalizeToUnitSphere(midpoint);
    rollCallMidpoints_.row(globalRollCallIndex) = midpoint.transpose();

    // Variables temporales para el roll call actual
    // Fortran: OLDZ, OLDD
    Eigen::VectorXd oldz = rollCallMidpoints_.row(globalRollCallIndex).transpose();
    Eigen::VectorXd oldd = rollCallSpreads_.row(globalRollCallIndex).transpose();
    CuttingPolarity polarity;

    // Obtener clasificacion inicial
    if (ns == 1)
    {
        // NS=1: Usar JAN11PT
        std::vector<double> projections(numVoters);
        for (int i = 0; i < numVoters; ++i)
        {
            projections[i] = coords(sortedIndices[i], 0);
        }

        double cuttingPoint = 0.0;
        double spread = 0.5;
        double accuracy1 = 0.0;
        double accuracy2 = 0.0;

        applyJan11pt(numVoters, projections, voteCodes,
                     cuttingPoint, spread, polarity,
                     accuracy1, accuracy2);

        // Primera iteracion: actualizar OLDZ, OLDD desde resultados
        if (iteration == config_.firstIteration)
        {
            // Normalizar punto de corte a [-1, 1]
            if (std::abs(cuttingPoint) > 1.0)
            {
                cuttingPoint = cuttingPoint / std::abs(cuttingPoint);
            }
            oldz(0) = cuttingPoint;
            oldd(0) = spread;
        }
    }
    else
    {
        // NS>1: Usar CUTPLANE
        applyCutplane(numVoters, coords, voteCodes, oldz, oldd, polarity);
    }

    // Guardar polaridad
    rollCallPolarity_[globalRollCallIndex] = polarity;

    // PROLLC2 + RCINT2: Optimizar parametros del roll call
    // Configuracion del optimizador de roll calls
    RollCallOptimizerConfig rcConfig;
    rcConfig.numOuterIterations = 5;
    rcConfig.numInnerIterations = 10;
    rcConfig.numSearchPoints = 25;

    // Ejecutar optimizacion
    // Fortran: CALL RCINT2(...)
    RollCallOptimizationResult rcResult = optimizeRollCall(
        legislatorCoords_,
        globalRollCallIndex,
        oldz,
        oldd,
        votes_,
        weights_,
        normalCDF_,
        rcConfig);

    // Actualizar parametros optimizados en el estado interno
    rollCallMidpoints_.row(globalRollCallIndex) = rcResult.midpoint.transpose();
    rollCallSpreads_.row(globalRollCallIndex) = rcResult.spread.transpose();

    // Actualizar contadores de clasificacion
    totalVotes += rcResult.totalVotes;
    classificationBefore += rcResult.totalVotes; // Aproximacion para antes
    classificationAfter += rcResult.correctClassified;
}

// Prepara datos para un roll call.
int DWNominate::prepareRollCallData(
    int congressIndex,
    int rollCallLocalIndex,
    int legislatorOffset,
    Eigen::MatrixXd &coords,
    std::vector<int> &voteCodes,
    std::vector<int> &sortedIndices,
    int &yesCount,
    int &noCount)
{
    int ns = config_.numDimensions;
    yesCount = 0;
    noCount = 0;

    // Encontrar el congreso en congressInfo_
    const CongressInfo *congressPtr = nullptr;
    for (const auto &c : congressInfo_)
    {
        if (c.index == congressIndex)
        {
            congressPtr = &c;
            break;
        }
    }

    if (!congressPtr)
    {
        return 0;
    }

    int numLegislatorsInCongress = congressPtr->numLegislators;
    int globalRollCall = congressPtr->rollCallOffset + rollCallLocalIndex;

    // Contar legisladores con voto valido
    std::vector<int> validLegislators;
    for (int i = 0; i < numLegislatorsInCongress; ++i)
    {
        int globalLeg = legislatorOffset + i;
        if (globalLeg >= static_cast<int>(votes_.getNumLegislators()))
        {
            continue;
        }
        if (globalRollCall >= static_cast<int>(votes_.getNumRollCalls()))
        {
            continue;
        }

        if (!votes_.isMissing(globalLeg, globalRollCall))
        {
            validLegislators.push_back(i);
            if (votes_.getVote(globalLeg, globalRollCall))
            {
                yesCount++;
            }
            else
            {
                noCount++;
            }
        }
    }

    int numVoters = static_cast<int>(validLegislators.size());
    if (numVoters == 0)
    {
        return 0;
    }

    // Extraer coordenadas y codigos de voto
    coords.resize(numVoters, ns);
    voteCodes.resize(numVoters);
    std::vector<double> projections(numVoters);

    for (int i = 0; i < numVoters; ++i)
    {
        int localLeg = validLegislators[i];
        int globalLeg = legislatorOffset + localLeg;

        for (int k = 0; k < ns; ++k)
        {
            coords(i, k) = legislatorCoords_(globalLeg, k);
        }

        // Voto: 1=Si, 6=No (codigos Fortran)
        if (votes_.getVote(globalLeg, globalRollCall))
        {
            voteCodes[i] = VoteCode::YES;
        }
        else
        {
            voteCodes[i] = VoteCode::NO;
        }

        projections[i] = coords(i, 0); // Primera dimension para ordenar
    }

    // Ordenar por primera dimension
    std::vector<size_t> rawIndices = argsort(projections);
    sortedIndices.resize(rawIndices.size());
    for (size_t i = 0; i < rawIndices.size(); ++i)
    {
        sortedIndices[i] = static_cast<int>(rawIndices[i]);
    }

    // Reordenar voteCodes segun el orden
    std::vector<int> sortedVoteCodes(numVoters);
    for (int i = 0; i < numVoters; ++i)
    {
        sortedVoteCodes[i] = voteCodes[sortedIndices[i]];
    }
    voteCodes = sortedVoteCodes;

    return numVoters;
}

// Aplica JAN11PT para NS=1.
void DWNominate::applyJan11pt(
    int numVoters,
    const std::vector<double> &projections,
    const std::vector<int> &voteCodes,
    double &cuttingPoint,
    double &spread,
    CuttingPolarity &polarity,
    double &accuracy1,
    double &accuracy2)
{
    // Probar ambas polaridades
    CuttingPolarity pol1(VoteCode::YES, VoteCode::NO);
    CuttingPolarity pol2(VoteCode::NO, VoteCode::YES);

    // Usar findCuttingPoint1DFixedPolarity de cutting_point.hpp
    CuttingPointResult result1 = findCuttingPoint1DFixedPolarity(
        projections, voteCodes, pol1);
    CuttingPointResult result2 = findCuttingPoint1DFixedPolarity(
        projections, voteCodes, pol2);

    accuracy1 = result1.counts.accuracy() * 100.0;
    accuracy2 = result2.counts.accuracy() * 100.0;

    // Seleccionar la mejor polaridad
    if (accuracy1 >= accuracy2)
    {
        polarity = pol1;
        cuttingPoint = result1.cuttingPoint;
        spread = 0.5; // Valor por defecto
    }
    else
    {
        polarity = pol2;
        cuttingPoint = result2.cuttingPoint;
        spread = -0.5;
    }
}

// Aplica CUTPLANE para NS>1.
void DWNominate::applyCutplane(
    int numVoters,
    const Eigen::MatrixXd &coords,
    const std::vector<int> &voteCodes,
    Eigen::VectorXd &midpoint,
    Eigen::VectorXd &spread,
    CuttingPolarity &polarity)
{
    int ns = config_.numDimensions;

    // Vector normal inicial
    Eigen::VectorXd normalVector = Eigen::VectorXd::Zero(ns);
    normalVector(0) = 1.0;

    // Llamar a classifyRollCall de cutting_plane.hpp
    bool searchEnabled = true; // IFIXX=1
    RollCallClassification result = classifyRollCall(
        coords, normalVector, voteCodes, searchEnabled);

    // Ajustar orientacion del vector normal
    Eigen::VectorXd zvec = normalVector;
    double ws = result.cuttingPoint;

    if (zvec(0) < 0.0)
    {
        zvec = -zvec;
        ws = -ws;
        polarity = CuttingPolarity(result.polarity.highSideVote,
                                   result.polarity.lowSideVote);
    }
    else
    {
        polarity = result.polarity;
    }

    // Calcular midpoint
    for (int k = 0; k < ns; ++k)
    {
        midpoint(k) = ws * zvec(k);
    }
    normalizeToUnitSphere(midpoint);

    // Calcular spread
    for (int k = 0; k < ns; ++k)
    {
        if (polarity.lowSideVote == VoteCode::YES)
        {
            spread(k) = 0.5 * zvec(k);
        }
        else
        {
            spread(k) = -0.5 * zvec(k);
        }
    }
}

// FASE DE LEGISLADORES
void DWNominate::executeLegislatorPhase()
{
    int uniqueCount = 0;

    // Loop sobre todos los IDs posibles
    for (int uniqueId = 0; uniqueId < static_cast<int>(legislatorPresence_.size());
         ++uniqueId)
    {

        const LegislatorPresence &presence = legislatorPresence_[uniqueId];

        // Verificar si este legislador tiene presencia en algun congreso
        if (presence.uniqueId < 0)
        {
            continue;
        }

        // Contar congresos en el rango
        int congressCount = 0;
        for (const auto &pair : presence.congressToDataIndex)
        {
            int congress = pair.first;
            if (congress >= config_.firstCongress &&
                congress <= config_.lastCongress)
            {
                congressCount++;
            }
        }

        if (congressCount == 0)
        {
            continue;
        }

        uniqueCount++;

        // Procesar legislador
        processLegislator(uniqueId, presence);
    }
    // Fin loop 48

    log("  Legisladores unicos procesados: " + std::to_string(uniqueCount));
}

/**
 * Procesa un legislador unico.
 */
void DWNominate::processLegislator(int uniqueId, const LegislatorPresence &presence)
{
    int ns = config_.numDimensions;

    // Construir LegislatorPeriodInfo desde presence
    LegislatorPeriodInfo periodInfo = buildLegislatorPeriodInfo(uniqueId, presence);

    // Determinar modelo temporal basado en numero de congresos
    int congressCount = presence.getNumCongresses();
    TemporalModel maxModel = TemporalModel::Constant;
    if (config_.temporalModel >= 1 && congressCount >= 5)
    {
        maxModel = TemporalModel::Linear;
    }
    if (config_.temporalModel >= 2 && congressCount >= 6)
    {
        maxModel = TemporalModel::Quadratic;
    }
    if (config_.temporalModel >= 3 && congressCount >= 7)
    {
        maxModel = TemporalModel::Cubic;
    }

    // Configuracion del optimizador
    LegislatorOptimizerConfig legConfig;
    legConfig.maxIterations = 10;
    legConfig.numSearchPointsConst = 25;
    legConfig.numSearchPointsTemporal = 10;

    // Ejecutar optimizacion
    LegislatorOptimizationResult legResult = optimizeLegislator(
        uniqueId,
        periodInfo,
        legislatorCoords_,
        rollCallMidpoints_,
        rollCallSpreads_,
        votes_,
        validRollCalls_,
        weights_,
        normalCDF_,
        maxModel,
        config_.firstCongress,
        config_.lastCongress,
        legConfig);

    // Reconstruir coordenadas en legislatorCoords_ desde coeficientes optimizados
    reconstructLegislatorCoords(presence, legResult.coefficients);

    // Actualizar varianzas

    if (config_.temporalModel == 0 || congressCount < 5)
    {
        // Modelo constante: XVAR(id, 1:6) desde OUTX0
        if (ns >= 1)
        {
            legislatorVariances_(uniqueId, 0) = legResult.covariance0(0, 0); // Var(X1)
        }
        if (ns >= 2)
        {
            legislatorVariances_(uniqueId, 3) = legResult.covariance0(1, 1); // Var(X2)
        }
        // Los terminos cruzados son 0 para modelo constante
        legislatorVariances_(uniqueId, 1) = 0.0;
        legislatorVariances_(uniqueId, 2) = 0.0;
        legislatorVariances_(uniqueId, 4) = 0.0;
        legislatorVariances_(uniqueId, 5) = 0.0;
    }
    else
    {
        // Modelo lineal o superior: XVAR desde OUTX1
        if (ns >= 1 && legResult.covariance1.rows() >= 2 * ns)
        {
            legislatorVariances_(uniqueId, 0) = legResult.covariance1(0, 0);
            legislatorVariances_(uniqueId, 1) = legResult.covariance1(ns, ns);
            legislatorVariances_(uniqueId, 2) = legResult.covariance1(0, ns);
        }
        if (ns >= 2 && legResult.covariance1.rows() >= 2 * ns)
        {
            legislatorVariances_(uniqueId, 3) = legResult.covariance1(1, 1);
            legislatorVariances_(uniqueId, 4) = legResult.covariance1(ns + 1, ns + 1);
            legislatorVariances_(uniqueId, 5) = legResult.covariance1(1, ns + 1);
        }
    }

    // Actualizar estadisticas por legislador
    for (const auto &pair : presence.congressToDataIndex)
    {
        int dataIndex = pair.second;
        if (dataIndex >= 0 && dataIndex < legislatorLogLikelihood_.rows())
        {
            legislatorLogLikelihood_(dataIndex, 0) = legResult.logLikelihood0;
            legislatorVoteCounts_(dataIndex, 0) = legResult.totalVotes;
        }
    }
}

// METODOS DE UTILIDAD
/**
 * Calcula log-likelihood global (PLOG).
 */
double DWNominate::computeLogLikelihood()
{
    // Construir vector de parametros de roll call
    std::vector<RollCallParameters> rollCallParams;
    int numRollCalls = static_cast<int>(rollCallMidpoints_.rows());

    for (int i = 0; i < numRollCalls; ++i)
    {
        RollCallParameters params(config_.numDimensions);
        params.midpoint = rollCallMidpoints_.row(i).transpose();
        params.spread = rollCallSpreads_.row(i).transpose();
        rollCallParams.push_back(params);
    }

    // Llamar a computeLogLikelihood de likelihood.hpp (funcion global)
    LikelihoodResult result = ::computeLogLikelihood(
        legislatorCoords_,
        rollCallParams,
        votes_,
        weights_,
        normalCDF_,
        validRollCalls_);

    // Actualizar estadisticas globales
    globalStats_ = result.stats;

    // Actualizar estadisticas por legislador
    for (int i = 0; i < static_cast<int>(result.legislatorLL.size()); ++i)
    {
        legislatorLogLikelihood_(i, 1) = result.legislatorLL[i];
        legislatorVoteCounts_(i, 1) = result.legislatorVotes[i];
    }

    return result.logLikelihood;
}

/**
 * Normaliza un vector a la esfera unitaria.
 */
void DWNominate::normalizeToUnitSphere(Eigen::VectorXd &point)
{
    double norm = point.norm();
    if (norm > 1.0)
    {
        point /= norm;
    }
}

/**
 * Verifica si un roll call es valido.
 */
bool DWNominate::isRollCallValid(int yesCount, int noCount) const
{
    int total = yesCount + noCount;
    if (total == 0)
    {
        return false;
    }
    int minority = std::min(yesCount, noCount);
    double margin = static_cast<double>(minority) / total;
    return margin >= config_.marginThreshold;
}

/**
 * Log de progreso.
 */
void DWNominate::log(const std::string &message) const
{
    if (config_.verbose)
    {
        std::cout << message << std::endl;
    }
}

// METODOS AUXILIARES DE INTEGRACION CON OPTIMIZADORES
/**
 * Construye vector de RollCallParameters desde estado interno.
 *
 * @return Vector de parametros de roll calls
 */
std::vector<RollCallParameters> DWNominate::buildRollCallParams() const
{
    std::vector<RollCallParameters> params;
    int numRollCalls = static_cast<int>(rollCallMidpoints_.rows());
    int ns = config_.numDimensions;

    params.reserve(numRollCalls);
    for (int i = 0; i < numRollCalls; ++i)
    {
        RollCallParameters rc(ns);
        rc.midpoint = rollCallMidpoints_.row(i).transpose();
        rc.spread = rollCallSpreads_.row(i).transpose();
        params.push_back(rc);
    }

    return params;
}

/**
 * Construye LegislatorPeriodInfo para un legislador.
 *
 * @param uniqueId ID unico del legislador
 * @param presence Informacion de presencia en congresos
 * @return Informacion de periodos para el optimizador
 */
LegislatorPeriodInfo DWNominate::buildLegislatorPeriodInfo(
    int uniqueId,
    const LegislatorPresence &presence) const
{
    // Determinar numero total de periodos
    int numPeriods = config_.lastCongress - config_.firstCongress + 1;
    LegislatorPeriodInfo info(numPeriods);

    // Llenar informacion para cada congreso donde sirvio el legislador
    for (const auto &pair : presence.congressToDataIndex)
    {
        int congress = pair.first;
        int dataIndex = pair.second;

        // Verificar que el congreso esta en el rango
        if (congress < config_.firstCongress || congress > config_.lastCongress)
        {
            continue;
        }

        // Convertir congreso a indice de periodo (0-based relativo a firstCongress)
        int periodIndex = congress - config_.firstCongress;

        // Obtener numero de roll calls para este congreso
        int numRollCalls = 0;
        for (const auto &cInfo : congressInfo_)
        {
            if (cInfo.index == congress)
            {
                numRollCalls = cInfo.numRollCalls;
                break;
            }
        }

        // Marcar como servido
        info.markServed(periodIndex, dataIndex, numRollCalls);
    }

    return info;
}

/**
 * Reconstruye coordenadas de legislador desde coeficientes temporales.
 * @param presence Informacion de presencia
 * @param coefficients Coeficientes temporales optimizados
 */
void DWNominate::reconstructLegislatorCoords(
    const LegislatorPresence &presence,
    const TemporalCoefficients &coefficients)
{
    int ns = config_.numDimensions;

    // Obtener lista de periodos servidos
    std::vector<int> servedPeriods;
    for (const auto &pair : presence.congressToDataIndex)
    {
        int congress = pair.first;
        if (congress >= config_.firstCongress && congress <= config_.lastCongress)
        {
            servedPeriods.push_back(congress);
        }
    }

    int kk = static_cast<int>(servedPeriods.size());
    if (kk == 0)
    {
        return;
    }

    // Calcular incremento temporal (igual que en buildLegendreTimeTrends)
    double xinc = 0.0;
    if (kk > 1)
    {
        xinc = 2.0 / (static_cast<double>(kk) - 1.0);
    }

    // Reconstruir coordenadas para cada periodo
    int periodIdx = 0;
    for (int congress : servedPeriods)
    {
        int dataIndex = presence.getDataIndex(congress);
        if (dataIndex < 0)
        {
            periodIdx++;
            continue;
        }

        // Calcular tiempo normalizado t en [-1, 1]
        double xtime = -1.0 + static_cast<double>(periodIdx) * xinc;

        // Calcular polinomios de Legendre
        double p0 = 1.0;
        double p1 = xtime;
        double p2 = (3.0 * xtime * xtime - 1.0) / 2.0;
        double p3 = (5.0 * xtime * xtime * xtime - 3.0 * xtime) / 2.0;

        // Reconstruir coordenadas para cada dimension
        for (int k = 0; k < ns; ++k)
        {
            double coord = coefficients(0, k) * p0 + coefficients(1, k) * p1 + coefficients(2, k) * p2 + coefficients(3, k) * p3;
            // Actualizar coordenada en la matriz de legisladores
            legislatorCoords_(dataIndex, k) = coord;
        }

        periodIdx++;
    }
}

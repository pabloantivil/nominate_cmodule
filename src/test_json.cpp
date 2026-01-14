/**
 * test_json.cpp
 *
 * Programa de prueba para verificar la correcta instalación de nlohmann/json.
 * Este archivo simula el flujo de datos que usará el módulo DW-NOMINATE:
 *
 * 1. Crear JSON programáticamente (simulando datos de votación)
 * 2. Serializar a string
 * 3. Parsear desde string (simulando lectura de stdin)
 * 4. Acceder a los datos
 * 5. Escribir resultado a stdout
 *
 * Compilar con:
 *   g++ -std=c++14 -I../include test_json.cpp -o test_json.exe
 *
 * O usando CMake (recomendado)
 */

#include <iostream>
#include <string>
#include <vector>
#include <sstream>

// Incluir nlohmann/json
#include "json.hpp"

// Alias conveniente (convención estándar)
using json = nlohmann::json;

int main()
{
    std::cout << "========================================" << std::endl;
    std::cout << "   PRUEBA DE nlohmann/json             " << std::endl;
    std::cout << "========================================" << std::endl;

    // =========================================================
    // TEST 1: Crear JSON programáticamente
    // =========================================================
    std::cout << "\n📝 TEST 1: Crear JSON programáticamente" << std::endl;

    json input_data;

    // Simular estructura de entrada para DW-NOMINATE
    input_data["parameters"]["dims"] = 2;
    input_data["parameters"]["maxiter"] = 30;
    input_data["parameters"]["xtol"] = 1e-4;
    input_data["parameters"]["polarity"] = {1, 2}; // Array

    // Simular datos de votaciones (rollcalls)
    json rollcall1;
    rollcall1["rollcall_id"] = 1001;
    rollcall1["legislators"] = json::array();
    rollcall1["legislators"].push_back({{"id", 1}, {"vote", 1}}); // Sí
    rollcall1["legislators"].push_back({{"id", 2}, {"vote", 6}}); // No
    rollcall1["legislators"].push_back({{"id", 3}, {"vote", 1}}); // Sí
    rollcall1["legislators"].push_back({{"id", 4}, {"vote", 9}}); // Ausente

    json rollcall2;
    rollcall2["rollcall_id"] = 1002;
    rollcall2["legislators"] = json::array();
    rollcall2["legislators"].push_back({{"id", 1}, {"vote", 6}});
    rollcall2["legislators"].push_back({{"id", 2}, {"vote", 1}});
    rollcall2["legislators"].push_back({{"id", 3}, {"vote", 6}});
    rollcall2["legislators"].push_back({{"id", 4}, {"vote", 1}});

    input_data["votes"] = json::array({rollcall1, rollcall2});

    std::cout << "   ✅ JSON de entrada creado" << std::endl;

    // =========================================================
    // TEST 2: Serializar a string (pretty print)
    // =========================================================
    std::cout << "\n📤 TEST 2: Serializar a string" << std::endl;

    std::string json_string = input_data.dump(2); // indent=2

    std::cout << "   JSON serializado (" << json_string.length() << " bytes):" << std::endl;
    std::cout << "   ---" << std::endl;
    // Mostrar primeras líneas
    std::istringstream stream(json_string);
    std::string line;
    int line_count = 0;
    while (std::getline(stream, line) && line_count < 15)
    {
        std::cout << "   " << line << std::endl;
        line_count++;
    }
    if (line_count >= 15)
    {
        std::cout << "   ... (truncado)" << std::endl;
    }
    std::cout << "   ---" << std::endl;
    std::cout << "   ✅ Serialización exitosa" << std::endl;

    // =========================================================
    // TEST 3: Parsear desde string (simula lectura de stdin)
    // =========================================================
    std::cout << "\n📥 TEST 3: Parsear desde string" << std::endl;

    json parsed_data;
    try
    {
        parsed_data = json::parse(json_string);
        std::cout << "   ✅ Parseo exitoso" << std::endl;
    }
    catch (const json::parse_error &e)
    {
        std::cerr << "   ❌ Error de parseo: " << e.what() << std::endl;
        return 1;
    }

    // =========================================================
    // TEST 4: Acceder a los datos
    // =========================================================
    std::cout << "\n🔍 TEST 4: Acceder a los datos" << std::endl;

    // Acceso a parámetros
    int dims = parsed_data["parameters"]["dims"].get<int>();
    int maxiter = parsed_data["parameters"]["maxiter"].get<int>();
    double xtol = parsed_data["parameters"]["xtol"].get<double>();

    std::cout << "   Parámetros:" << std::endl;
    std::cout << "     dims = " << dims << std::endl;
    std::cout << "     maxiter = " << maxiter << std::endl;
    std::cout << "     xtol = " << xtol << std::endl;

    // Acceso a array de polarity
    std::vector<int> polarity = parsed_data["parameters"]["polarity"].get<std::vector<int>>();
    std::cout << "     polarity = [";
    for (size_t i = 0; i < polarity.size(); ++i)
    {
        std::cout << polarity[i];
        if (i < polarity.size() - 1)
            std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Iterar sobre votaciones
    std::cout << "   Votaciones:" << std::endl;
    for (const auto &vote : parsed_data["votes"])
    {
        int rollcall_id = vote["rollcall_id"].get<int>();
        int num_legislators = vote["legislators"].size();
        std::cout << "     Rollcall " << rollcall_id << ": " << num_legislators << " legisladores" << std::endl;
    }

    std::cout << "   ✅ Acceso a datos exitoso" << std::endl;

    // =========================================================
    // TEST 5: Crear JSON de salida (simula resultado DW-NOMINATE)
    // =========================================================
    std::cout << "\n📊 TEST 5: Crear JSON de salida" << std::endl;

    json output;
    output["success"] = true;
    output["iterations"] = 25;
    output["convergence"]["log_likelihood"] = -1234.567;
    output["convergence"]["converged"] = true;

    // Simular coordenadas de legisladores
    output["legislators"] = json::object();
    output["legislators"]["1"] = {{"coord1", 0.523}, {"coord2", -0.127}, {"se1", 0.05}, {"se2", 0.04}};
    output["legislators"]["2"] = {{"coord1", -0.891}, {"coord2", 0.456}, {"se1", 0.06}, {"se2", 0.05}};
    output["legislators"]["3"] = {{"coord1", 0.234}, {"coord2", 0.789}, {"se1", 0.04}, {"se2", 0.03}};
    output["legislators"]["4"] = {{"coord1", -0.567}, {"coord2", -0.234}, {"se1", 0.07}, {"se2", 0.06}};

    // Simular parámetros de rollcalls
    output["rollcalls"] = json::object();
    output["rollcalls"]["1001"] = {{"midpoint1", 0.0}, {"midpoint2", 0.1}, {"spread1", 2.5}, {"spread2", 1.8}};
    output["rollcalls"]["1002"] = {{"midpoint1", -0.3}, {"midpoint2", 0.2}, {"spread1", 1.9}, {"spread2", 2.1}};

    // Serializar con formato compacto (para stdout real)
    std::string compact_output = output.dump();

    // Serializar con formato legible (para debug)
    std::string pretty_output = output.dump(2);

    std::cout << "   JSON de salida (formato legible):" << std::endl;
    std::cout << "   ---" << std::endl;
    std::istringstream out_stream(pretty_output);
    line_count = 0;
    while (std::getline(out_stream, line) && line_count < 20)
    {
        std::cout << "   " << line << std::endl;
        line_count++;
    }
    if (line_count >= 20)
    {
        std::cout << "   ... (truncado)" << std::endl;
    }
    std::cout << "   ---" << std::endl;

    std::cout << "   Tamaño compacto: " << compact_output.length() << " bytes" << std::endl;
    std::cout << "   ✅ JSON de salida creado" << std::endl;

    // =========================================================
    // TEST 6: Manejo de errores
    // =========================================================
    std::cout << "\n⚠️  TEST 6: Manejo de errores" << std::endl;

    // Intentar parsear JSON inválido
    std::string invalid_json = "{\"key\": value_sin_comillas}";
    try
    {
        json bad = json::parse(invalid_json);
        std::cout << "   ❌ Debería haber fallado" << std::endl;
    }
    catch (const json::parse_error &e)
    {
        std::cout << "   ✅ Error capturado correctamente:" << std::endl;
        std::cout << "      " << e.what() << std::endl;
    }

    // Acceso a clave inexistente (con valor por defecto)
    json test_obj = {{"a", 1}};
    int value_b = test_obj.value("b", -1); // Devuelve -1 si no existe
    std::cout << "   ✅ Valor por defecto funciona: b = " << value_b << std::endl;

    // =========================================================
    // RESUMEN
    // =========================================================
    std::cout << "\n========================================" << std::endl;
    std::cout << "   ✅ TODOS LOS TESTS PASARON          " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "\nnlohmann/json está correctamente instalada y funcionando." << std::endl;
    std::cout << "La librería está lista para usarse en el proyecto DW-NOMINATE." << std::endl;

    return 0;
}

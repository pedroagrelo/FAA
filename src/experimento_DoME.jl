using CSV, DataFrames
include("D:/CopiaPedro/CLASE/2º/2ºcuatri/Fundamentos de Aprendizaje Automático/Práctica2aParte/FAA/src/P1_soluciones.jl")
data = CSV.read("alzheimers_limpio_balanced.csv", DataFrame)

# 2. Extraer entradas (X) y salida (y), y normalizar las entradas 
inputs = Matrix(data[:, Not("Diagnosis")])
targets = data[:, "Diagnosis"]
inputs = normalizeMinMax(inputs)

# 4. Convertir las etiquetas a strings para DoME
targets = string.(targets)

# Cargar índices de validación cruzada
indicesCV = CSV.read("indices_crossval.csv", DataFrame).Fold

# -------------------------------
# 6. Valores de nodos a probar
# -------------------------------
nodos_a_probar = [5, 10, 15, 20, 25, 30, 35, 40]

#Inicializar tabla de resultados
resultados_dome = DataFrame(
    MaxNodes = Int[],
    AccuracyMean = Float64[], AccuracyStd = Float64[],
    F1Mean = Float64[], F1Std = Float64[],
    RecallMean = Float64[], RecallStd = Float64[],
    PrecisionMean = Float64[], PrecisionStd = Float64[],
    SpecificityMean = Float64[], SpecificityStd = Float64[],
    NPVMean = Float64[], NPVStd = Float64[]
)

# 7. Ejecutar experimentos
println("=== EXPERIMENTO DoME - Alzheimer ===")
for max_nodes in nodos_a_probar
    println("\n[DoME] Máximo nodos = ", max_nodes)
    
    acc,
    _, 
    recall,
    specificity,
    precision,
    npv,
    f1,
    _ = modelCrossValidation(:DoME,

        Dict("maximumNodes" => max_nodes),
        (inputs, targets),
        indicesCV)

    # Guardar resultados en DataFrame
    push!(resultados_dome, (
        max_nodes,
        mean(acc), std(acc),
        mean(f1), std(f1),
        mean(recall), std(recall),
        mean(precision), std(precision),
        mean(specificity), std(specificity),
        mean(npv), std(npv)
    ))
    
    println("Resultados para maximumNodes = ", max_nodes)
    println("→ Accuracy     : ", round(mean(acc), digits=4), " ± ", round(std(acc), digits=4))
    println("→ F1 Score     : ", round(mean(f1), digits=4), " ± ", round(std(f1), digits=4))
    println("→ Recall       : ", round(mean(recall), digits=4), " ± ", round(std(recall), digits=4))
    println("→ Precision    : ", round(mean(precision), digits=4), " ± ", round(std(precision), digits=4))
    println("→ Specificity  : ", round(mean(specificity), digits=4), " ± ", round(std(specificity), digits=4))
    println("→ NPV          : ", round(mean(npv), digits=4), " ± ", round(std(npv), digits=4))
    #println("→ Tiempo total : ", round(tiempo_total, digits=2), " segundos")  #SOBRA

end

# 6. Guardar CSV 
CSV.write("resultados_crossval_dome.csv", resultados_dome)
println("Resultados guardados en 'resultados_crossval_dome.csv'")
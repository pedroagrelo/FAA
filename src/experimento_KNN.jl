using CSV, DataFrames, Statistics
include("D:/CopiaPedro/CLASE/2º/2ºcuatri/Fundamentos de Aprendizaje Automático/Práctica2aParte/FAA/src/P1_soluciones.jl")

# 1. Cargar datos
data = CSV.read("alzheimers_limpio_balanced.csv", DataFrame)
inputs = Matrix(data[:, Not("Diagnosis")])
targets = string.(data[:, "Diagnosis"])
inputs = normalizeMinMax(inputs)

# 2. Cargar índices de validación cruzada
indicesCV = CSV.read("indices_crossval.csv", DataFrame).Fold

# 3. Valores de k a probar
k_valores = [1, 3, 5, 7, 9, 11, 13, 15]

# 4. Inicializar DataFrame de resultados
resultados_knn = DataFrame(
    K = Int[],
    AccuracyMean = Float64[], AccuracyStd = Float64[],
    F1Mean = Float64[], F1Std = Float64[],
    RecallMean = Float64[], RecallStd = Float64[],
    PrecisionMean = Float64[], PrecisionStd = Float64[],
    SpecificityMean = Float64[], SpecificityStd = Float64[],
    NPVMean = Float64[], NPVStd = Float64[]
)

# 5. Ejecutar experimentos
for k in k_valores
    println("\n[KNN] Vecinos = $k")

    acc, _, recall, specificity, precision, npv, f1, _ =
        modelCrossValidation(
            :KNeighborsClassifier,
            Dict("n_neighbors" => k),
            (inputs, targets),
            indicesCV
        )

    # Guardar en DataFrame
    push!(resultados_knn, (
        k,
        mean(acc), std(acc),
        mean(f1), std(f1),
        mean(recall), std(recall),
        mean(precision), std(precision),
        mean(specificity), std(specificity),
        mean(npv), std(npv)
    ))

    # Mostrar por pantalla
    println("→ Accuracy     : ", round(mean(acc), digits=4), " ± ", round(std(acc), digits=4))
    println("→ F1 Score     : ", round(mean(f1), digits=4), " ± ", round(std(f1), digits=4))
    println("→ Recall       : ", round(mean(recall), digits=4), " ± ", round(std(recall), digits=4))
    println("→ Precision    : ", round(mean(precision), digits=4), " ± ", round(std(precision), digits=4))
    println("→ Specificity  : ", round(mean(specificity), digits=4), " ± ", round(std(specificity), digits=4))
    println("→ NPV          : ", round(mean(npv), digits=4), " ± ", round(std(npv), digits=4))
end

# 6. Guardar resultados en CSV
CSV.write("resultados_crossval_knn.csv", resultados_knn)
println("Resultados guardados en 'resultados_crossval_knn.csv'")

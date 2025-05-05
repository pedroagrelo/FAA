using CSV, DataFrames, Random, Statistics
include("P1_soluciones.jl")  # Aquí están las funciones como crossvalidation y normalizeMinMax
using DecisionTree

# ------------------------------------------
# Normalización Min-Max
# ------------------------------------------
function normalizacionDecisionTree(inputs::Matrix{<:Real})
    return normalizeMinMax(inputs)  # Reutiliza tu función de P1
end

# ------------------------------------------
# Ejecutar Árbol de Decisión con validación cruzada k=10
# ------------------------------------------
function ejecutarDecisionTree()

    # 1. Leer dataset limpio
    df = CSV.read("alzheimers_limpio_balanced.csv", DataFrame)
    inputs = Matrix(select(df, Not(:Diagnosis)))
    targets = Vector(df.Diagnosis)

    # 2. Normalizar entradas
    X_norm = normalizacionDecisionTree(inputs)

    # 3. Generar índices de validación cruzada estratificada
    k = 10
    cv_indices = crossvalidation(targets, k)

    # 4. Profundidades a evaluar
    profundidades = [2, 4, 6, 8, 10, 12]

    resultados = DataFrame(
        Profundidad = Int[],
        Accuracy = Float64[],
        F1_Score = Float64[],
        Precision = Float64[],
        Recall = Float64[],
        Specificity = Float64[],
        NPV = Float64[],
        Tiempo = Float64[]
    )

    for profundidad in profundidades
        println("\nEvaluando árbol con profundidad = ", profundidad)

        tiempo = @elapsed begin
            (acc, _), (_, _), (recall, _), (specificity, _), (precision, _), (npv, _), (f1, _), _ =
                modelCrossValidation(:DecisionTreeClassifier,
                    Dict("max_depth" => profundidad),
                    (X_norm, targets), cv_indices)
        end

        push!(resultados, (
            profundidad,
            round(acc, digits=4),
            round(f1, digits=4),
            round(precision, digits=4),
            round(recall, digits=4),
            round(specificity, digits=4),
            round(npv, digits=4),
            round(tiempo, digits=2)
        ))
    end

    println("\nResumen resultados Árbol de Decisión:")
    show(resultados, allcols=true)

    CSV.write("resultados_crossval_dt.csv", resultados)

end

# Ejecutar
ejecutarDecisionTree()

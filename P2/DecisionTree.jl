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

    # Resultados almacenando todos los valores de cada fold
    resultados = DataFrame(
        Profundidad = Int[],
        Accuracy_vals = Vector{Float64}[],
        F1_Score_vals = Vector{Float64}[],
        Precision_vals = Vector{Float64}[],
        Recall_vals = Vector{Float64}[],
        Specificity_vals = Vector{Float64}[],
        NPV_vals = Vector{Float64}[],
        Tiempo_vals = Vector{Float64}[]
    )

    for profundidad in profundidades
        println("\nEvaluando árbol con profundidad = ", profundidad)

        # Inicializamos los vectores para cada métrica
        accuracy_vals = Float64[]
        f1_vals = Float64[]
        precision_vals = Float64[]
        recall_vals = Float64[]
        specificity_vals = Float64[]
        npv_vals = Float64[]
        tiempo_vals = Float64[]

        for fold in 1:k
            # Aquí realizamos la validación cruzada para cada fold
            tiempo = @elapsed begin
                (acc, _), (_, _), (recall, _), (specificity, _), (precision, _), (npv, _), (f1, _), _ =
                    modelCrossValidation(:DecisionTreeClassifier,
                        Dict("max_depth" => profundidad),
                        (X_norm, targets), cv_indices)
            end

            # Almacenamos los valores en los vectores correspondientes
            push!(accuracy_vals, round(acc, digits=4))
            push!(f1_vals, round(f1, digits=4))
            push!(precision_vals, round(precision, digits=4))
            push!(recall_vals, round(recall, digits=4))
            push!(specificity_vals, round(specificity, digits=4))
            push!(npv_vals, round(npv, digits=4))
            push!(tiempo_vals, round(tiempo, digits=2))
        end

        # Ahora agregamos los vectores a la tabla de resultados
        push!(resultados, (
            profundidad,
            accuracy_vals,
            f1_vals,
            precision_vals,
            recall_vals,
            specificity_vals,
            npv_vals,
            tiempo_vals
        ))
    end

    println("\nResumen resultados Árbol de Decisión:")
    show(resultados, allcols=true)

    # Guardar los resultados completos (sin promedios) en un CSV
    CSV.write("resultados_crossval_dt.csv", resultados)

end

# Ejecutar
ejecutarDecisionTree()

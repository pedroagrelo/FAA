module experimentoDT

using CSV, DataFrames, DecisionTree, Statistics, HypothesisTests
include("P1_soluciones.jl") # Debe contener modelCrossValidation sin data leakage

function ejecutarDecisionTree()
    # Carga de datos
    df = CSV.read("P2/OTROS_ARCHIVOS/alzheimers_limpio_balanced.csv", DataFrame)
    inputs = Matrix(select(df, Not(:Diagnosis)))
    targets = Vector(df.Diagnosis)

    # Cargar índices de validación cruzada comunes
    cv_indices = CSV.read("P2/OTROS_ARCHIVOS/indices_crossval.csv", DataFrame).Fold

    # Profundidades a evaluar
    profundidades = [2, 4, 6, 8, 10, 12]

    # DataFrame para almacenar resultados
    resultados = DataFrame(
        Profundidad = Int[],
        Accuracy = String[],
        F1_Score = String[],
        Precision = String[],
        Recall = String[],
        Specificity = String[],
        NPV = String[],
        Tiempo = Float64[]
    )

    for profundidad in profundidades
        println("Profundidad: $profundidad")
        tiempo = @elapsed begin
            accs, _, recalls, specificities, precisions, npvs, f1s, _ =
                modelCrossValidation(:DecisionTreeClassifier,
                    Dict("max_depth" => profundidad),
                    (inputs, targets), cv_indices)  # SIN normalizar aquí
        end

        push!(resultados, (
            profundidad,
            string(accs),
            string(f1s),
            string(precisions),
            string(recalls),
            string(specificities),
            string(npvs),
            round(tiempo, digits=2)
        ))
    end

    CSV.write("P2/RESULTADOS/resultados_crossval_dt.csv", resultados)

    # Test ANOVA solo para Accuracy
    realizar_anova(resultados)
end

# Función para realizar el test ANOVA sobre Accuracy
function realizar_anova(df::DataFrame, colname::Symbol = :Accuracy)
    expanded = expand_metric_column(df, colname)
    grupos = [expanded[expanded.Profundidad .== profundidad, :Valor] for profundidad in unique(expanded.Profundidad)]
    anova_result = OneWayANOVATest(grupos...)
    println("Resultados del test ANOVA para Accuracy: ")
    println(anova_result)
end

# Función para expandir una columna tipo String con métricas por fold
function expand_metric_column(df::DataFrame, colname::Symbol)
    expanded_data = DataFrame(Profundidad = Int[], Fold = Int[], Valor = Float64[])
    for (i, row) in enumerate(eachrow(df))
        metric_values = parse.(Float64, split(row[colname], r"[ \[\],]+")[2:end-1])
        for (j, val) in enumerate(metric_values)
            push!(expanded_data, (df.Profundidad[i], j, val))
        end
    end
    return expanded_data
end

end # module

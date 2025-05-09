module experimentoDT

using CSV, DataFrames, DecisionTree, Statistics, HypothesisTests, Distributions
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
                    (inputs, targets), cv_indices)
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

    # Realizar ANOVA + Tukey
    realizar_anova(resultados)
     # Seleccionar y mostrar la mejor profundidad según el accuracy
    seleccionar_mejor_profundidad(resultados)
end

# Función para realizar el test ANOVA sobre Accuracy
function realizar_anova(df::DataFrame, colname::Symbol = :Accuracy)
    expanded = expand_metric_column(df, colname)
    grupos = [expanded[expanded.Profundidad .== profundidad, :Valor] for profundidad in unique(expanded.Profundidad)]
    anova_result = OneWayANOVATest(grupos...)
    println("Resultados del test ANOVA para Accuracy:")
    println(anova_result)

    if pvalue(anova_result) < 0.05
        println("→ Se rechaza H₀ con un p-valor de $(round(pvalue(anova_result), digits=4)). Se realizarán comparaciones múltiples.")
        realizar_tukey_hsd(df, colname)
    else
        println("→ No se rechaza H₀. No se hacen comparaciones múltiples.")
    end
end

# Función para comparaciones múltiples estilo Tukey HSD
function realizar_tukey_hsd(df::DataFrame, colname::Symbol = :Accuracy)
    println("\nComparaciones múltiples tipo Tukey HSD:")

    expanded = expand_metric_column(df, colname)
    grupos = unique(expanded.Profundidad)
    α = 0.05

    for i = 1:length(grupos)-1
        for j = i+1:length(grupos)
            g1, g2 = grupos[i], grupos[j]
            vals1 = expanded[expanded.Profundidad .== g1, :Valor]
            vals2 = expanded[expanded.Profundidad .== g2, :Valor]

            diff = mean(vals1) - mean(vals2)
            pooled_var = (var(vals1) + var(vals2)) / 2
            se = sqrt(pooled_var * (1/length(vals1) + 1/length(vals2)))
            t_stat = abs(diff) / se

            dfree = length(vals1) + length(vals2) - 2
            critical_t = quantile(TDist(dfree), 1 - α/2)
            significant = t_stat > critical_t
            resultado = significant ? "DIFERENCIA SIGNIFICATIVA" : "sin diferencia"

            println("Comparación $g1 vs $g2: t = $(round(t_stat, digits=3)) (umbral = $(round(critical_t, digits=3))) → $resultado")
        end
    end
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

# Función para seleccionar la mejor profundidad según los resultados de Accuracy
function seleccionar_mejor_profundidad(df::DataFrame)
    # Encuentra la profundidad con el mejor accuracy promedio
    mejor_profundidad = argmax(df.Accuracy)[1]
    mejor_accuracy = df.Accuracy[mejor_profundidad]

    println("\nLa mejor profundidad es: ", df.Profundidad[mejor_profundidad])
    println("Con un accuracy de: ", mejor_accuracy)
end

end # module

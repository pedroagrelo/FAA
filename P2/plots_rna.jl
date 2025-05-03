using CSV, DataFrames, Plots, StatsBase, HypothesisTests
include("experimentos.jl")


function resumir_metricas(df::DataFrame)
    return DataFrame(
        Arquitectura = df.Arquitectura,
        Accuracy_mean = mean.(df.AccuracyMean),
        Accuracy_std = std.(df.AccuracyMean),
        F1_mean = mean.(df.F1Mean),
        F1_std = std.(df.F1Std),
        Precision_mean = mean.(df.PrecisionMean),
        Precision_std = std.(df.PrecisionStd),
        Recall_mean = mean.(df.RecallMean),
        Recall_std = std.(df.RecallStd),
    )
end

function graficar_metricas_desde_csv(archivo_csv::String)
    df = CSV.read(archivo_csv, DataFrame)

    # Convertir arquitectura a string más limpio (sin corchetes)
    etiquetas = replace.(df.Arquitectura, r"\[|\]" => "")

    # Accuracy
    bar(
        etiquetas, df.Accuracy_mean;
        yerror = df.Accuracy_std,
        ylabel = "Accuracy",
        xlabel = "Arquitectura",
        title = "Accuracy (± std)",
        rotation = 45,
        legend = false,
        size = (700, 500)
    )
    savefig("accuracy_rna.png")

    # F1
    bar(
        etiquetas, df.F1_mean;
        yerror = df.F1_std,
        ylabel = "F1-score",
        xlabel = "Arquitectura",
        title = "F1-score (± std)",
        rotation = 45,
        legend = false,
        size = (700, 500)
    )
    savefig("f1_rna.png")

    # Precision
    bar(
        etiquetas, df.Precision_mean;
        yerror = df.Precision_std,
        ylabel = "Precision",
        xlabel = "Arquitectura",
        title = "Precisión (± std)",
        rotation = 45,
        legend = false,
        size = (700, 500)
    )
    savefig("precision_rna.png")

    # Recall
    bar(
        etiquetas, df.Recall_mean;
        yerror = df.Recall_std,
        ylabel = "Recall",
        xlabel = "Arquitectura",
        title = "Recall (± std)",
        rotation = 45,
        legend = false,
        size = (700, 500)
    )
    savefig("recall_rna.png")

    println("\n  Gráficas guardadas como PNG.")
end


function graficar_matriz_confusion(matriz::Matrix{Int64})
    p = heatmap(
        matriz,
        c = :blues,
        xlabel = "Predicción",
        ylabel = "Valor real",
        xticks = ([1, 2], ["No Alzheimer", "Alzheimer"]),
        yticks = ([1, 2], ["No Alzheimer", "Alzheimer"]),
        title = "Matriz de Confusión Global (RNA)",
        size = (500, 400),
        # annotate = true,
        colorbar = false
    )

      # Etiquetas correspondientes a cada celda
      etiquetas = [["TN", "FP"],
      ["FN", "TP"]]

    # Añadir anotaciones a la matriz
    # Añadir anotaciones con valor y etiqueta
    for i in 1:2
        for j in 1:2
            texto = "$(matriz[i, j]) $(etiquetas[i][j])"
            annotate!(p, j, i, text(texto, :white, 12, halign=:center, valign=:center))
        end
    end
 

    savefig(p, "confusion_rna.png")
    println(" Matriz de confusión guardada como 'confusion_rna.png'")
end

# 1. Ejecutar experimentos
# resultados, conf_matrix_final = ejecutarRNA()


# 2. Resumir métricas por fold
resumen = resumir_metricas(resultados)

# 3. Guardar el resumen en un CSV
CSV.write("resumen_resultados_crossval_rna.csv", resumen)

# 4. Graficar desde CSV
graficar_metricas_desde_csv("resumen_resultados_crossval_rna.csv")

# Suponiendo que ya tienes el DataFrame `df` con las métricas
# Cargar el dataset de resultados
# df = CSV.read("resultados_crossval_rna.csv", DataFrame)
# anova_resultados = realizar_anova(df)

# Imprimir los resultados del test ANOVA
# println("Resultados del test ANOVA: ")
# println(anova_resultados)

df = CSV.read("resultados_crossval_rna.csv", DataFrame)
anova_results = realizar_anova(df, :AccuracyMean)
println(anova_results)
#Como es un promedio de los k folds, la matriz de confusión es necesario redondearla para ajustarse a un entero
conf_matrix_final = round.(Int64, conf_matrix_final)
# Guardar la matriz de confusión global como una imagen
graficar_matriz_confusion(conf_matrix_final)

println(conf_matrix_final)

#[622 138; 157 603]

# Verdaderos Negativos (No Alzheimer, No Alzheimer): 622

# Falsos Positivos (No Alzheimer, Alzheimer): 138

# Falsos Negativos (Alzheimer, No Alzheimer): 157

# Verdaderos Positivos (Alzheimer, Alzheimer): 603


# # 1. Generar matriz global a partir de arquitectura elegida
# conf_total = obtener_matriz_confusion_global(
#     [18, 15],  # arquitectura ganadora
#     X_norm,
#     targets,
#     cv_indices
# )

# # 2. Graficar la matriz
# graficar_matriz_confusion_global(conf_total)


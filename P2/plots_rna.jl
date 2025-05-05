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
        Specificity_mean = mean.(df.SpecificityMean),
        Specificity_std = std.(df.SpecificityStd),
        NPV_mean = mean.(df.NPVMean),
        NPV_std = std.(df.NPVStd)
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

     # Specificity
     bar(
        etiquetas, df.Specificity_mean;
        yerror = df.Specificity_std,
        ylabel = "Specificity",
        xlabel = "Arquitectura",
        title = "Specificity (± std)",
        rotation = 45,
        legend = false,
        size = (700, 500)
    )
    savefig("specificity_rna.png")

     # NPV
     bar(
        etiquetas, df.NPV_mean;
        yerror = df.NPV_std,
        ylabel = "NPV",
        xlabel = "Arquitectura",
        title = "NPV (± std)",
        rotation = 45,
        legend = false,
        size = (700, 500)
    )
    savefig("NPV_rna.png")




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

#1. Ejecutar experimentos
resultados, conf_matrix_final = ejecutarRNA()

# 2. Resumir métricas por fold
resumen = resumir_metricas(resultados)

# 3. Guardar el resumen en un CSV
CSV.write("resumen_resultados_crossval_rna.csv", resumen)

# 4. Graficar desde CSV
graficar_metricas_desde_csv("resumen_resultados_crossval_rna.csv")


# Imprimir los resultados del test ANOVA
println("Resultados del test ANOVA: ")
df_anova = CSV.read("resultados_crossval_rna.csv", DataFrame)
anova_results = realizar_anova(df_anova, :AccuracyMean)
println(anova_results)




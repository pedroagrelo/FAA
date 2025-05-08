module plotsRNA

using CSV, DataFrames, Plots, StatsBase, HypothesisTests

export resumir_metricas_RNA, graficar_metricas_RNA 

function resumir_metricas_RNA(df::DataFrame)
    return DataFrame(
        Arquitectura = df.Arquitectura,
        Accuracy_mean = mean.(df.AccuracyMean),
        Accuracy_std = std.(df.AccuracyMean),
        F1_mean = mean.(df.F1Mean),
        F1_std = std.(df.F1Mean),
        Precision_mean = mean.(df.PrecisionMean),
        Precision_std = std.(df.PrecisionMean),
        Recall_mean = mean.(df.RecallMean),
        Recall_std = std.(df.RecallMean),
        Specificity_mean = mean.(df.SpecificityMean),
        Specificity_std = std.(df.SpecificityMean),
        NPV_mean = mean.(df.NPVMean),
        NPV_std = std.(df.NPVMean)
    )
end

function graficar_metricas_RNA(archivo_csv::String)
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
    savefig("P2/IMÁGENES/accuracy_rna.png")

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
    savefig("P2/IMÁGENES/f1_rna.png")

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
    savefig("P2/IMÁGENES/precision_rna.png")

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
    savefig("P2/IMÁGENES/recall_rna.png")

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
    savefig("P2/IMÁGENES/specificity_rna.png")

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
    savefig("P2/IMÁGENES/NPV_rna.png")




    println("\n  Gráficas guardadas como PNG.")
end

end

# #1. Ejecutar experimentos
# resultados, conf_matrix_final = ejecutarRNA()

# # 2. Resumir métricas por fold
# resumen = resumir_metricas(resultados)

# # 3. Guardar el resumen en un CSV
# CSV.write("resumen_resultados_crossval_rna.csv", resumen)

# # 4. Graficar desde CSV
# graficar_metricas_desde_csv("resumen_resultados_crossval_rna.csv")


# # Imprimir los resultados del test ANOVA
# println("Resultados del test ANOVA: ")
# df_anova = CSV.read("resultados_crossval_rna.csv", DataFrame)
# anova_results = realizar_anova(df_anova, :AccuracyMean)
# println(anova_results)




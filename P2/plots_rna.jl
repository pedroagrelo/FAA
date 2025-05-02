using CSV, DataFrames, Plots
include("experimentos.jl")

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
        size = (700, 400)
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
        size = (700, 400)
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
        size = (700, 400)
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
        size = (700, 400)
    )
    savefig("recall_rna.png")

    println("\n 📈 Gráficas guardadas como PNG.")
end


function graficar_matriz_confusion(matriz::Matrix{Int})
    heatmap(
        matriz,
        c = :blues,
        xlabel = "Predicción",
        ylabel = "Valor real",
        xticks = ([1, 2], ["No Alzheimer", "Alzheimer"]),
        yticks = ([1, 2], ["No Alzheimer", "Alzheimer"]),
        title = "Matriz de Confusión Global (RNA)",
        size = (500, 400),
        annotate = true,
        colorbar = false
    )
    savefig("confusion_rna.png")
    println("✅ Matriz de confusión guardada como 'confusion_rna.png'")
end

# 1. Ejecutar experimentos
resultados, conf_matrix_final = ejecutarRNA()

# 2. Graficar desde CSV
graficar_metricas_desde_csv("resultados_crossval_rna.csv")

# Guardar la matriz de confusión global como una imagen
graficar_matriz_confusion(conf_matrix_final)


# # 1. Generar matriz global a partir de arquitectura elegida
# conf_total = obtener_matriz_confusion_global(
#     [18, 15],  # arquitectura ganadora
#     X_norm,
#     targets,
#     cv_indices
# )

# # 2. Graficar la matriz
# graficar_matriz_confusion_global(conf_total)

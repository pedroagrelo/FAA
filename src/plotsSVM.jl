module plotsSVM

using CSV, DataFrames, Plots, StatsBase

export resumir_metricas_svm, graficar_metricas_svm

# -----------------------------------------------
# Resumen estadístico para gráficas de SVM
# -----------------------------------------------
function resumir_metricas_svm(df::DataFrame)
    grouped = groupby(df, :Configuracion)
    return combine(grouped, 
        :Kernel => first => :kernel,
        :C => first,
        :Gamma => first,
        :Degree => first,
        :Accuracy => mean => :Accuracy_mean,
        :Accuracy => std => :Accuracy_std,
        :F1_Score => mean => :F1_mean,
        :F1_Score => std => :F1_std,
        :Tiempo => mean => :Tiempo
    )
end


# ------------------------
# Gráficas de barras 
# ------------------------
function graficar_metricas_svm(archivo_csv::String)
    df = CSV.read(archivo_csv, DataFrame)
    etiquetas = df.Configuracion

    # Accuracy
    bar(
        etiquetas, df.Accuracy_mean;
        yerror = df.Accuracy_std,
        ylabel = "Accuracy",
        xlabel = "Configuración SVM",
        title = "Accuracy (± std)",
        rotation = 45,
        legend = false,
        size = (800, 500)
    )
    savefig("accuracy_svm.png")

    # F1-score
    bar(
        etiquetas, df.F1_mean;
        yerror = df.F1_std,
        ylabel = "F1-score",
        xlabel = "Configuración SVM",
        title = "F1-score (± std)",
        rotation = 45,
        legend = false,
        size = (800, 500)
    )
    savefig("f1_svm.png")

    println("\nGráficas de SVM guardadas como PNG.")
end

end

module plotsSVM

using CSV, DataFrames, Plots, StatsBase, JSON

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
        :Precision => mean => :Precision_mean,
        :Precision => std => :Precision_std,
        :Recall => mean => :Recall_mean,
        :Recall => std => :Recall_std,
        :Specificity => mean => :Specificity_mean,
        :Specificity => std => :Specificity_std,
        :VPN => mean => :VPN_mean,
        :VPN => std => :VPN_std,
        :Tiempo => mean => :Tiempo
    )
end


function parse_vector_columns!(df::DataFrame, cols::Vector{Symbol})
    for col in cols
        df[!, col] = [JSON.parse(row) for row in df[!, col]]
    end
    return df
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
    savefig("P2/IMÁGENES/accuracy_svm.png")

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
    savefig("P2/IMÁGENES/f1_svm.png")


    # Precision
    bar(
        etiquetas, df.Precision_mean;
        yerror = df.Precision_std,
        ylabel = "Precision",
        xlabel = "Configuración SVM",
        title = "Precision (± std)",
        rotation = 45,
        legend = false,
        size = (800, 500)
    )
    savefig("P2/IMÁGENES/precision_svm.png")

    # Recall
    bar(
        etiquetas, df.Recall_mean;
        yerror = df.Recall_std,
        ylabel = "Recall",
        xlabel = "Configuración SVM",
        title = "Recall (± std)",
        rotation = 45,
        legend = false,
        size = (800, 500)
    )
    savefig("P2/IMÁGENES/recall_svm.png")

    # Specificity
    bar(
        etiquetas, df.Specificity_mean;
        yerror = df.Specificity_std,
        ylabel = "Specificity",
        xlabel = "Configuración SVM",
        title = "Specificity (± std)",
        rotation = 45,
        legend = false,
        size = (800, 500)
    )
    savefig("P2/IMÁGENES/specificity_svm.png")

    # VPN
    bar(
        etiquetas, df.VPN_mean;
        yerror = df.VPN_std,
        ylabel = "VPN",
        xlabel = "Configuración SVM",
        title = "VPN (± std)",
        rotation = 45,
        legend = false,
        size = (800, 500)
    )
    savefig("P2/IMÁGENES/vpn_svm.png")

    println("\nGráficas de SVM guardadas como PNG.")
end

end

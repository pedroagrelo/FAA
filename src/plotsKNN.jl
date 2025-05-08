module plotsKNN

using CSV, DataFrames, Plots

export graficar_metricas_knn

function graficar_metricas_knn(archivo_csv::String)
    df = CSV.read(archivo_csv, DataFrame)

    # Convertir K a string para el eje X
    etiquetas = string.(df.K)

    # Lista de métricas a graficar
    metricas = ["Accuracy", "F1", "Precision", "Recall", "Specificity", "NPV"]

    for metrica in metricas
        mean_col = Symbol(metrica * "Mean")
        std_col  = Symbol(metrica * "Std")

        bar(
            etiquetas,
            df[!, mean_col],
            yerror = df[!, std_col],
            xlabel = "Número de vecinos (k)",
            ylabel = metrica,
            title = "$metrica (± std) - KNN",
            legend = false,
            size = (800, 500)
        )
        savefig("P2/IMÁGENES/$metrica _knn.png")
    end

    println("Gráficas guardadas como PNG para KNN.")
end

end

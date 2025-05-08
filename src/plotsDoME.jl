module plotsDoME

using CSV, DataFrames, Plots

export graficar_metricas_dome

function graficar_metricas_dome(archivo_csv::String)
    df = CSV.read(archivo_csv, DataFrame)

    # Convertir número de nodos a string (eje x)
    etiquetas = string.(df.MaxNodes)

    # Lista de métricas a graficar
    metricas = ["Accuracy", "F1", "Precision", "Recall", "Specificity", "NPV"]

    for metrica in metricas
        mean_col = Symbol(metrica * "Mean")
        std_col  = Symbol(metrica * "Std")

        bar(
            etiquetas,
            df[!, mean_col],
            yerror = df[!, std_col],
            xlabel = "Nodos máximos (DoME)",
            ylabel = metrica,
            title = "$metrica (± std) - DoME",
            legend = false,
            size = (800, 500)
        )
        savefig("P2/IMÁGENES/$metrica _dome.png")
    end

    println("\n✅ Gráficas generadas como PNG para DoME.")
end

end

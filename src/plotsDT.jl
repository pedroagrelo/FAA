module plotsDT

using CSV, DataFrames, Statistics, Plots

function parse_vector_string(s::AbstractString)
    try
        return eval(Meta.parse(s))
    catch
        println("Error al convertir a vector: ", s)
        return [NaN]
    end
end

function resumir_metricas_dt_detalle(archivo::String)
    df = CSV.read(archivo, DataFrame)
    for col in names(df)
        if eltype(df[!, col]) <: AbstractString && occursin("[", df[1, col])
            df[!, col] = parse_vector_string.(df[!, col])
        end
    end

    return DataFrame(
        Profundidad = df.Profundidad,
        Accuracy_mean = mean.(df.Accuracy),
        Accuracy_std = std.(df.Accuracy),
        F1_mean = mean.(df.F1_Score),
        F1_std = std.(df.F1_Score),
        Precision_mean = mean.(df.Precision),
        Precision_std = std.(df.Precision),
        Recall_mean = mean.(df.Recall),
        Recall_std = std.(df.Recall),
        Specificity_mean = mean.(df.Specificity),
        Specificity_std = std.(df.Specificity),
        NPV_mean = mean.(df.NPV),
        NPV_std = std.(df.NPV)
    )
end

function graficar_metricas_barras_dt(archivo_resumen::String)
    df = CSV.read(archivo_resumen, DataFrame)
    profundidades = string.(df.Profundidad)
    metricas = ["Accuracy", "F1", "Precision", "Recall", "Specificity", "NPV"]

    for metrica in metricas
        mean_col = Symbol(metrica * "_mean")
        std_col  = Symbol(metrica * "_std")

        bar(
            profundidades,
            df[!, mean_col],
            yerror = df[!, std_col],
            legend = false,
            title = "Comparativa de $metrica por profundidad",
            xlabel = "Profundidad del árbol",
            ylabel = metrica,
            color = :steelblue,
            size = (800, 500)
        )
        savefig("P2/IMÁGENES/$metrica DT.png")
    end
end

end # module

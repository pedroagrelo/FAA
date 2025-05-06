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

    # Detectar automáticamente columnas que contienen vectores
    for col in names(df)
        # Solo convertimos si los valores parecen listas
        if eltype(df[!, col]) <: AbstractString && occursin("[", df[1, col])
            df[!, col] = parse_vector_string.(df[!, col])
        end
    end

    resumen = DataFrame(
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

    return resumen
end

using CSV, DataFrames, Plots, Statistics

function graficar_metricas_barras_dt(archivo_resumen::String)
    df = CSV.read(archivo_resumen, DataFrame)
    profundidades = string.(df.Profundidad)  # Convertimos a string para etiquetas de eje X
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

        savefig("grafica_$metrica.png")
    end
end


resumen = resumir_metricas_dt_detalle("resultados_crossval_dt.csv")
CSV.write("resumen_resultados_crossval_dt.csv", resumen)
graficar_metricas_barras_dt("resumen_resultados_crossval_dt.csv")

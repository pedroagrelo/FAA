using CSV, DataFrames, Statistics, Plots

function parse_vector_string(s::AbstractString)
    return eval(Meta.parse(s))
end

function resumir_metricas_dt_detalle(archivo::String)
    df = CSV.read(archivo, DataFrame)

    # Convertir strings a vectores reales
    for col in names(df)
        if endswith(col, "_vals")
            df[!, col] = parse_vector_string.(df[!, col])
        end
    end

    resumen = DataFrame(
        Profundidad = df.Profundidad,
        Accuracy_mean = mean.(df.Accuracy_vals),
        Accuracy_std = std.(df.Accuracy_vals),
        F1_mean = mean.(df.F1_Score_vals),
        F1_std = std.(df.F1_Score_vals),
        Precision_mean = mean.(df.Precision_vals),
        Precision_std = std.(df.Precision_vals),
        Recall_mean = mean.(df.Recall_vals),
        Recall_std = std.(df.Recall_vals),
        Specificity_mean = mean.(df.Specificity_vals),
        Specificity_std = std.(df.Specificity_vals),
        NPV_mean = mean.(df.NPV_vals),
        NPV_std = std.(df.NPV_vals)
    )

    return resumen
end

resumen = resumir_metricas_dt_detalle("resultados_crossval_dt.csv")
CSV.write("resumen_resultados_crossval_dt.csv", resumen)
graficar_metricas_dt("resumen_resultados_crossval_dt.csv")




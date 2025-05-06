module Preprocesamiento

using CSV
using DataFrames
using Statistics
using Plots
using Random

export calcular_correlaciones, filtrar_dataset, balancear_dataset, analizar_outliers


function calcular_correlaciones(path::String, output_plot_path::String)
    # 1. Leer el archivo CSV
    df = CSV.read(path, DataFrame)

    # 2. Seleccionar solo columnas numéricas (incluyendo Diagnosis)
    numeric_df = select(df, names(df, eltype.(eachcol(df)) .<: Number))

    # 3. Separar Diagnosis como vector
    target = numeric_df[:, :Diagnosis]

    # 4. Calcular correlación de Pearson entre Diagnosis y cada otra variable
    cor_vals = [cor(target, numeric_df[:, col]) for col in names(numeric_df) if col != :Diagnosis]
    colnames = [col for col in names(numeric_df) if col != :Diagnosis]

    # 5. Graficar como heatmap 1D (barra vertical de correlaciones)
    bar(
        reverse(cor_vals);
        orientation = :horizontal,
        yticks = (1:length(colnames), reverse(colnames)),
        title = "Correlación con Diagnosis (Pearson)",
        xlabel = "Correlación",
        color = :coolwarm,
        xlim = (-1, 1),
        legend = false,
        size = (800, 600)
    )

    # 7. Guardar figura
    savefig(output_plot_path)
    println("¡Hecho! Correlación con Diagnosis guardada como imagen.")
end

function filtrar_dataset(path::String, output_path::String)
    df = CSV.read(path, DataFrame)

    selected_features = [
        "MMSE", "FunctionalAssessment", "MemoryComplaints",
        "BehavioralProblems", "ADL", "Diagnosis"
    ]

    filtered_df = select(df, selected_features)
    CSV.write(output_path, filtered_df)

    return filtered_df
end

# Realiza balanceo por subsampling de la clase mayoritaria
function balancear_dataset(input_path::String, output_path::String)
    df = CSV.read(input_path, DataFrame)

    df_con = filter(row -> row.Diagnosis == 1, df)
    df_sin = filter(row -> row.Diagnosis == 0, df)

    n_minor = nrow(df_con)
    Random.seed!(42)
    idx_sin_sub = sample(1:nrow(df_sin), n_minor; replace = false)
    df_sin_sub = df_sin[idx_sin_sub, :]

    df_balanced = vcat(df_con, df_sin_sub)
    df_balanced = df_balanced[shuffle(1:nrow(df_balanced)), :]

    println("Distribución tras balanceo: ", countmap(df_balanced.Diagnosis))
    CSV.write(output_path, df_balanced)
    println("Dataset balanceado guardado en '$output_path'.")
    return df_balanced
end

# Detecta columnas numéricas no categóricas
function columnas_numericas_no_categoricas(df::DataFrame; umbral_unicos::Int = 5)
    return [col for col in names(df) if eltype(df[!, col]) <: Number && length(unique(skipmissing(df[!, col]))) > umbral_unicos]
end

# Analiza outliers usando IQR y muestra resultados
function analizar_outliers(csv_path::String)
    df = CSV.read(csv_path, DataFrame)
    columnas_validas = columnas_numericas_no_categoricas(df)
    println("Análisis de outliers por columna (IQR, sin categóricas):\n")

    total_outliers = 0
    for col in columnas_validas
        datos = skipmissing(df[!, col])
        q1 = quantile(datos, 0.25)
        q3 = quantile(datos, 0.75)
        iqr = q3 - q1
        lim_inf = q1 - 1.1 * iqr
        lim_sup = q3 + 1.1 * iqr
        outliers = sum(x -> x < lim_inf || x > lim_sup, datos)
        total_outliers += outliers
        println(rpad(col, 35), "→ ", outliers)
    end

    if total_outliers == 0
        println("\n No se han detectado outliers en el dataset.")
    else
        println("\n Total de outliers detectados: ", total_outliers)
    end

end

end
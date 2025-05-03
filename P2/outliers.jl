using CSV, DataFrames, Statistics

# Función para detectar columnas numéricas no categóricas
function columnas_numericas_no_categoricas(df::DataFrame; umbral_unicos::Int = 5)
    return [col for col in names(df) if eltype(df[!, col]) <: Number && length(unique(df[!, col])) > umbral_unicos]
end

# Función para contar outliers usando IQR en columnas numéricas no categóricas
function contar_outliers_iqr(df::DataFrame)
    columnas_validas = columnas_numericas_no_categoricas(df)
    println("Outliers por columna (usando IQR, excluyendo categóricas):\n")
    for col in columnas_validas
        datos = skipmissing(df[!, col])
        q1 = quantile(datos, 0.25)
        q3 = quantile(datos, 0.75)
        iqr = q3 - q1
        lim_inf = q1 - 1.1 * iqr
        lim_sup = q3 + 1.1 * iqr
        outliers = sum(x -> x < lim_inf || x > lim_sup, datos)
        println(rpad(col, 35), "→ ", outliers)
    end
end


# Carga tu dataset aquí (cambia la ruta si es necesario)
df = CSV.read("P2/alzheimers_disease_data.csv", DataFrame)

# Ejecuta el análisis
contar_outliers_iqr(df)

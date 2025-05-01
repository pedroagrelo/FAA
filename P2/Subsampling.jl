using CSV
using DataFrames
using Random
using StatsBase   # Si no lo tienes, instala con: import Pkg; Pkg.add("StatsBase")

# 1. Leer el dataset
df = CSV.read("P2/alzheimers_limpio.csv", DataFrame)

# 2. Separar clases
df_con  = filter(row -> row.Diagnosis == 1, df)  # Con Alzheimer
df_sin  = filter(row -> row.Diagnosis == 0, df)  # Sin Alzheimer

# 3. Submuestreo de la clase mayoritaria
n_minor = nrow(df_con)
Random.seed!(42)  # Para reproducibilidad
idx_sin_sub = sample(1:nrow(df_sin), n_minor; replace = false)
df_sin_sub   = df_sin[idx_sin_sub, :]

# 4. Combinar y barajar
df_balanced = vcat(df_con, df_sin_sub)
df_balanced = df_balanced[shuffle(1:nrow(df_balanced)), :]


# 6. Comprobar el balance
println(countmap(df_balanced.Diagnosis))

# 7. Guardar a CSV
CSV.write("alzheimers_limpio_balanced.csv", df_balanced)

println("Subsampling balanceado completado y guardado en 'alzheimers_balanced.csv'.")
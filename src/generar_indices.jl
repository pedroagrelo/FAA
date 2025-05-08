using CSV, DataFrames
include("P1_soluciones.jl")  # donde está tu función crossvalidation

# Cargar datos
data = CSV.read("alzheimers_limpio_balanced.csv", DataFrame)
targets = data.Diagnosis

# Generar índices estratificados
k = 10
indicesCV = crossvalidation(targets, k)

# Guardar en archivo CSV
df_indices = DataFrame(Fold = indicesCV)
CSV.write("indices_crossval.csv", df_indices)

println("Índices guardados en 'indices_crossval.csv'")
using CSV, DataFrames
include("P1_soluciones.jl")  # donde está tu función crossvalidation

# Cargar datos
data = CSV.read("alzheimers_limpio_balanced.csv", DataFrame)
targets = data.Diagnosis

# Generar índices estratificados
k = 10
indicesCV = crossvalidation(targets, k)

# Guardar en archivo CSV
df_indices = DataFrame(Fold = indicesCV)
CSV.write("indices_crossval.csv", df_indices)

println("Índices guardados en 'indices_crossval.csv'")

using CSV, DataFrames
include("P1_soluciones.jl")  # donde está tu función crossvalidation

# Cargar datos
data = CSV.read("P2/OTROS ARCHIVOS/alzheimers_limpio_balanced.csv", DataFrame)
targets = data.Diagnosis

# Generar índices estratificados
k = 10
indicesCV = crossvalidation(length(targets), k)

# Guardar en archivo CSV
df_indices = DataFrame(Fold = indicesCV)
CSV.write("P2/OTROS ARCHIVOS/indices_crossval2.csv", df_indices)

println("Índices guardados en 'indices_crossval2.csv'")

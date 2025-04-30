using CSV, DataFrames, CategoricalArrays, MLJ

# Cargar dataset
df = CSV.read("alzheimers_disease_data.csv", DataFrame)
first(df, 5)  # ver las primeras 5 filas
# Mostrar tipos de datos por columna
eltypes = eltype.(eachcol(df))
println(eltypes)
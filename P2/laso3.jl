using CSV
using DataFrames
using GLMNet
using Statistics

# === Cargar datos ===
df = CSV.read("P2/alzheimers_disease_data.csv", DataFrame)

# === Variable objetivo ===
y = convert(Vector{Float64}, df.Diagnosis)

# === Columnas numéricas ===
numeric_cols = names(df, eltype.(eachcol(df)) .<: Number)

# Eliminar la variable objetivo del conjunto de entrada
feature_cols = setdiff(numeric_cols, ["Diagnosis"])

# Extraer solo las columnas numéricas relevantes
X = Matrix{Float64}(df[:, feature_cols])

# === Eliminar columnas con varianza cero ===
stds = std(X, dims=1)
nonzero_var_idx = vec(stds .> 0)
X = X[:, nonzero_var_idx]
feature_cols = feature_cols[nonzero_var_idx]

# === Normalización ===
X_mean = mean(X, dims=1)
X_std = std(X, dims=1)
X_normalized = (X .- X_mean) ./ X_std

# === Ajuste Lasso ===
fit = glmnet(X_normalized, y, alpha=1.0)

# === Validación cruzada ===
cv = glmnetcv(X_normalized, y, alpha=1.0)

# Selección del mejor lambda
best_lambda_index = argmin(cv.meanloss)
best_lambda = cv.lambda[best_lambda_index]

# === Obtener coeficientes con el mejor lambda ===
best_coef = cv.path.betas[:, best_lambda_index]

# === Mostrar información ===
println("Columnas usadas como features:")
println(feature_cols)
println("Número de columnas: ", length(feature_cols))
println("Longitud de coeficientes (sin intercepto): ", length(best_coef))

# === Mostrar variables seleccionadas por Lasso ===
println("Variables seleccionadas por Lasso (coef ≠ 0):")
for (i, name) in enumerate(feature_cols)
    coef_val = best_coef[i]  # Los coeficientes no incluyen el intercepto en 'path.betas'
    if abs(coef_val) > 1e-6
        println("  $name → coef = $(round(coef_val, digits=4))")
    end
end

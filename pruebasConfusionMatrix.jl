

#FUNCIÓN 1
# Datos de ejemplo (matrices de 3 clases)
outputs = Bool[1 0 0; 1 0 0; 1 0 0; 0 1 0; 0 1 0; 0 1 0; 0 0 1; 0 0 1; 0 0 1]
targets = Bool[1 0 0; 0 1 0; 0 0 1; 1 0 0; 0 1 0; 0 0 1; 1 0 0; 0 1 0; 0 0 1]

# Ejecutar la primera función (la que ya tienes)
(acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = confusionMatrix(outputs, targets, weighted=true)

# Los asserts que ya funcionan deberían pasar
@assert(isapprox(acc, 1/3.) && isapprox(errorRate, 2/3.) && isapprox(recall, 1/3.) && 
        isapprox(specificity, 2/3.) && isapprox(precision, 1/3.) && isapprox(NPV, 2/3.) && 
        isapprox(F1, 1/3.) && confMatrix == [1 1 1; 1 1 1; 1 1 1])



#FUNCIÓN 2
include("firmas.jl")
# Generamos una matriz de valores reales como outputs
outputs_real = Float64[0.6 0.2 0.2; 0.5 0.3 0.2; 0.4 0.4 0.2; 
                       0.3 0.5 0.2; 0.2 0.6 0.2; 0.1 0.7 0.2; 
                       0.2 0.2 0.6; 0.3 0.3 0.4; 0.4 0.4 0.2]

# Definimos las salidas deseadas (targets) como una matriz de booleanos
targets = Bool[1 0 0; 1 0 0; 1 0 0; 
               0 1 0; 0 1 0; 0 1 0; 
               0 0 1; 0 0 1; 0 0 1]
# Ejecutar la segunda función (con umbral por defecto 0.5)
(acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = confusionMatrix(outputs_real, targets, weighted=true)

println("Accuracy: ", acc)
println("Error Rate: ", errorRate)
println("Recall: ", recall)
println("Specificity: ", specificity)
println("Precision: ", precision)
println("NPV: ", NPV)
println("F1: ", F1)
println("Confusion Matrix: ", confMatrix)

# Los asserts que ya funcionan deberían pasar
@assert(isapprox(acc, 1/3.) && isapprox(errorRate, 2/3.) && isapprox(recall, 1/3.) && 
        isapprox(specificity, 2/3.) && isapprox(precision, 1/3.) && isapprox(NPV, 2/3.) && 
        isapprox(F1, 1/3.) && confMatrix == [1 1 1; 1 1 1; 1 1 1])
        




#FUNCIÓN 3
include("firmas.jl")
# Generar un conjunto de clases (por ejemplo, "A", "B", "C")
classes = ["A", "B", "C"]

# Generar outputs y targets como arrays de cualquier tipo (por ejemplo, strings)
outputs_any = ["A", "B", "C", "A", "B", "C", "A", "B", "C"]
targets_any = ["A", "B", "C", "A", "B", "C", "A", "B", "C"]

# Ejecutar la tercera función
(acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = confusionMatrix(outputs_any, targets_any, classes, weighted=true)

# Los asserts que ya funcionan deberían pasar
@assert(isapprox(acc, 1/3.) && isapprox(errorRate, 2/3.) && isapprox(recall, 1/3.) && 
        isapprox(specificity, 2/3.) && isapprox(precision, 1/3.) && isapprox(NPV, 2/3.) && 
        isapprox(F1, 1/3.) && confMatrix == [1 1 1; 1 1 1; 1 1 1])



        
#FUNCIÓN 4
include("firmas.jl")
# Generar outputs y targets como arrays de cualquier tipo (por ejemplo, números)
outputs = Bool[1 0 1; 0 1 0; 1 0 1; 0 1 0; 0 1 0; 0 1 0; 0 0 1; 0 0 1; 0 0 1]
targets = Bool[1 0 1; 0 1 0; 1 0 1; 0 1 0; 0 1 0; 0 1 0; 0 0 1; 0 0 1; 0 0 1]


# Ejecutar la cuarta función
(acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = confusionMatrix(outputs, targets, weighted=true)

# Los asserts que ya funcionan deberían pasar
@assert(isapprox(acc, 1/3.) && isapprox(errorRate, 2/3.) && isapprox(recall, 1/3.) && 
        isapprox(specificity, 2/3.) && isapprox(precision, 1/3.) && isapprox(NPV, 2/3.) && 
        isapprox(F1, 1/3.) && confMatrix == [1 1 1; 1 1 1; 1 1 1])




include("firmas.jl")
# Prueba 1: Clasificación binaria con valores booleanos
println("Prueba 1: Clasificación binaria con valores booleanos")
outputs = sin.(1:8) .>= 0
targets = [falses(4); trues(4)]
(acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = confusionMatrix(outputs, targets)
println("Accuracy: ", acc)
println("Error Rate: ", errorRate)
println("Recall: ", recall)
println("Specificity: ", specificity)
println("Precision: ", precision)
println("NPV: ", NPV)
println("F1: ", F1)
println("Matriz de Confusión:")
println(confMatrix)
@assert(isapprox(acc, 0.375) && isapprox(errorRate, 1-0.375) && isapprox(recall, 0.5) && isapprox(specificity, 0.25) && isapprox(precision, 0.4) && isapprox(NPV, 1/3.) && isapprox(F1, 4/9.) && confMatrix==[1 3; 2 2])
println("Prueba 1 pasada correctamente.\n")

# Prueba 2: Clasificación binaria con valores reales y umbral
println("Prueba 2: Clasificación binaria con valores reales y umbral")
outputs = sin.(1:8)
targets = [falses(4); trues(4)]
(acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = confusionMatrix(outputs, targets; threshold=0.9)
println("Accuracy: ", acc)
println("Error Rate: ", errorRate)
println("Recall: ", recall)
println("Specificity: ", specificity)
println("Precision: ", precision)
println("NPV: ", NPV)
println("F1: ", F1)
println("Matriz de Confusión:")
println(confMatrix)
@assert(isapprox(acc, 0.5) && isapprox(errorRate, 0.5) && isapprox(recall, 0.25) && isapprox(specificity, 0.75) && isapprox(precision, 0.5) && isapprox(NPV, 0.5) && isapprox(F1, 1/3.) && confMatrix==[3 1; 3 1])
println("Prueba 2 pasada correctamente.\n")

# Prueba 3: Clasificación multiclase con valores booleanos y weighted=true
println("Prueba 3: Clasificación multiclase con valores booleanos y weighted=true")
outputs = Float64[1 0 0; 1 0 0; 1 0 0; 0 1 0; 0 1 0; 0 1 0; 0 0 1; 0 0 1; 0 0 1].+ 1
targets = Bool[1 0 0; 0 1 0; 0 0 1; 1 0 0; 0 1 0; 0 0 1; 1 0 0; 0 1 0; 0 0 1]
(acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = confusionMatrix(outputs, targets; weighted=true)
println("Accuracy: ", acc)
println("Error Rate: ", errorRate)
println("Recall: ", recall)
println("Specificity: ", specificity)
println("Precision: ", precision)
println("NPV: ", NPV)
println("F1: ", F1)
println("Matriz de Confusión:")
println(confMatrix)
@assert(isapprox(acc, 1/3.) && isapprox(errorRate, 2/3.) && isapprox(recall, 1/3.) && isapprox(specificity, 2/3.) && isapprox(precision, 1/3.) && isapprox(NPV, 2/3.) && isapprox(F1, 1/3.) && confMatrix==[1 1 1; 1 1 1; 1 1 1])
println("Prueba 3 pasada correctamente.\n")

# Prueba 4: Clasificación multiclase con valores categóricos
println("Prueba 4: Clasificación multiclase con valores categóricos")
targets = repeat([1, 2, 3], 50)
outputs = repeat(unique(targets), 50)
(acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = confusionMatrix(outputs, targets)
println("Accuracy: ", acc)
println("Error Rate: ", errorRate)
println("Recall: ", recall)
println("Specificity: ", specificity)
println("Precision: ", precision)
println("NPV: ", NPV)
println("F1: ", F1)
println("Matriz de Confusión:")
println(confMatrix)
@assert(isapprox(acc, 1/3.) && isapprox(errorRate, 2/3.) && isapprox(recall, 1/3.) && isapprox(specificity, 2/3.) && isapprox(precision, 1/3.) && isapprox(NPV, 2/3.) && isapprox(F1, 1/3.) && confMatrix==[17 17 16; 17 16 17; 16 17 17])
println("Prueba 4 pasada correctamente.\n")

println("¡Todas las pruebas se han ejecutado correctamente!")

# Función de prueba
function test_confusionMatrix()
        # Crear datos de prueba
        outputs = Float64[0.6 0.2 0.7; 0.4 0.3 0.8; 0.5 0.5 0.9; 0.1 0.7 0.6; 0.6 0.5 0.4]  # Salidas continuas
        targets = Bool[1 0 1; 0 1 1; 1 0 1; 0 1 1; 1 1 0]  # Objetivos reales binarios
    
        # Llamar a la función de clasificación binaria
        threshold = 0.5  # Umbral de conversión
        (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = confusionMatrix(outputs, targets, threshold=threshold, weighted=true)
        println("outputs_bool: ", outputs_bool) 
        # Imprimir los resultados
        println("Accuracy: ", acc)
        println("Error Rate: ", errorRate)
        println("Recall: ", recall)
        println("Specificity: ", specificity)
        println("Precision: ", precision)
        println("NPV: ", NPV)
        println("F1: ", F1)
        println("Confusion Matrix: ", confMatrix)
    
        # Validar los resultados esperados con un assert
        @assert isapprox(acc, 0.6)
        @assert isapprox(errorRate, 0.4)
        @assert isapprox(recall, 0.6)
        @assert isapprox(specificity, 0.8)
        @assert isapprox(precision, 0.75)
        @assert isapprox(NPV, 0.75)
        @assert isapprox(F1, 0.6667)
        @assert confMatrix == [2 1 0; 1 3 1; 1 1 3]  # Aquí asumiendo que se genera esta matriz
    
        println("Test passed!")
    end
    
    # Ejecutar la prueba
    test_confusionMatrix()
    

include("49202488Z_49472510Q_35592754R_32726079T.jl")
    # Ejemplo de prueba
outputs = Float64[1 0 0; 1 0 0; 1 0 0; 0 1 0; 0 1 0; 0 1 0; 0 0 1; 0 0 1; 0 0 1] .+ 1
targets = Bool[1 0 0; 0 1 0; 0 0 1; 1 0 0; 0 1 0; 0 0 1; 1 0 0; 0 1 0; 0 0 1]
threshold = 0.5

# Llamar la función para ver cómo se genera outputs_bool
confusionMatrix(outputs, targets; threshold=threshold)

using Random

# Generar datos de prueba para clasificación binaria
outputs_binary = rand(10, 1)  # Probabilidades entre 0 y 1
targets_binary = rand(Bool, 10, 1)  # Etiquetas verdaderas como valores booleanos

# Generar datos de prueba para clasificación multiclase con 3 clases
outputs_multiclass = rand(10, 3)  # Probabilidades de salida
targets_multiclass = falses(10, 3)
for i in 1:10
    targets_multiclass[i, rand(1:3)] = true  # Asignar una clase aleatoria como verdadera
end

# Prueba con datos binarios
println("Test clasificación binaria:")
result_bin = confusionMatrix(outputs_binary, targets_binary; threshold=0.5, weighted=true)
println("Resultado binario: ", result_bin)

# Prueba con datos multiclase
println("\nTest clasificación multiclase:")
result_multi = confusionMatrix(outputs_multiclass, targets_multiclass; threshold=0.5, weighted=true)
println("Resultado multiclase: ", result_multi)







using Random

# Función auxiliar para imprimir matrices de forma más clara
function printMatrix(label, matrix)
    println(label, ":")
    println(matrix)
end

# Generar datos de prueba
Random.seed!(42)  # Fijamos semilla para reproducibilidad

## 🔹 1. Prueba de clasificación binaria con diferentes thresholds
outputs_binary = rand(10, 1)  # Probabilidades entre 0 y 1
targets_binary = rand(Bool, 10, 1)  # Etiquetas verdaderas (booleanas)

println("\n🔹 Test clasificación binaria (threshold = 0.5):")
result_bin_05 = confusionMatrix(outputs_binary, targets_binary; threshold=0.5, weighted=true)
println("Resultado: ", result_bin_05)

println("\n🔹 Test clasificación binaria (threshold = 0.7):")
result_bin_07 = confusionMatrix(outputs_binary, targets_binary; threshold=0.7, weighted=true)
println("Resultado: ", result_bin_07)

## 🔹 2. Prueba de clasificación multiclase con 3 clases
outputs_multiclass = rand(10, 3)  # Probabilidades de salida (10 muestras, 3 clases)
targets_multiclass = falses(10, 3)
for i in 1:10
    targets_multiclass[i, rand(1:3)] = true  # Cada muestra tiene una clase activa
end

println("\n🔹 Test clasificación multiclase (3 clases, threshold = 0.5):")
result_multi_3 = confusionMatrix(outputs_multiclass, targets_multiclass; threshold=0.5, weighted=true)
println("Resultado: ", result_multi_3)

## 🔹 3. Caso límite: todas las salidas son `true`
outputs_all_true = ones(Bool, 10, 3)
targets_all_true = ones(Bool, 10, 3)

println("\n🔹 Test clasificación multiclase (todo true):")
result_all_true = confusionMatrix(outputs_all_true, targets_all_true; weighted=true)
println("Resultado: ", result_all_true)

## 🔹 4. Caso límite: todas las salidas son `false`
outputs_all_false = zeros(Bool, 10, 3)
targets_all_false = zeros(Bool, 10, 3)

println("\n🔹 Test clasificación multiclase (todo false):")
result_all_false = confusionMatrix(outputs_all_false, targets_all_false; weighted=true)
println("Resultado: ", result_all_false)

## 🔹 5. Caso con clases desbalanceadas
outputs_imbalanced = rand(10, 3) .> 0.8  # La mayoría de valores serán `false`
targets_imbalanced = falses(10, 3)
for i in 1:8
    targets_imbalanced[i, 1] = true  # La primera clase tiene más ejemplos
end
for i in 9:10
    targets_imbalanced[i, rand(2:3)] = true  # Clases minoritarias

println("\n🔹 Test clasificación multiclase (clases desbalanceadas):")
result_imbalanced = confusionMatrix(outputs_imbalanced, targets_imbalanced; weighted=true)
println("Resultado: ", result_imbalanced)
end



include("49202488Z_49472510Q_35592754R_32726079T.jl")
# Prueba con una matriz booleana de una sola columna
outputs_bool = Bool[1; 0; 1; 0; 1]  # Predicciones
targets_bool = Bool[1; 0; 0; 1; 1]  # Valores reales

# Convertimos a matriz con una sola columna
outputs_bool_matrix = reshape(outputs_bool, :, 1)
targets_bool_matrix = reshape(targets_bool, :, 1)

# Ejecutamos la función corregida
resultados = confusionMatrix(outputs_bool_matrix, targets_bool_matrix)
println("Resultados con matriz de una sola columna:", resultados)

# Prueba con valores reales y umbral personalizado
outputs_real = [0.8; 0.3; 0.6; 0.2; 0.9]  # Probabilidades
targets_bool = Bool[1; 0; 0; 1; 1]  # Valores reales

# Convertimos a matriz con una sola columna
outputs_real_matrix = reshape(outputs_real, :, 1)
targets_bool_matrix = reshape(targets_bool, :, 1)

# Probamos con un umbral distinto
threshold = 0.7
resultados_umbral = confusionMatrix(outputs_real_matrix, targets_bool_matrix; threshold=threshold)
println("Resultados con umbral personalizado:", resultados_umbral)


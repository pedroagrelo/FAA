

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





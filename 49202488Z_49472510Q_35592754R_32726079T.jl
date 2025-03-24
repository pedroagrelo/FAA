# Tened en cuenta que en este archivo todas las funciones tienen puesta la palabra reservada 'function' y 'end' al final
# Según cómo las defináis, podrían tener que llevarlas o no

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 2 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Statistics
using Flux
using Flux.Losses

function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    num_classes = length(classes)
    num_samples = length(feature)
    
    if num_classes <= 2
        # Caso binario: Generar un vector booleano y convertirlo en una matriz columna
        return reshape(feature .== classes[1], num_samples, 1) #NUM SAMPLE FILAS 1 columna
    else
        # Caso multiclase: Crear una matriz de valores booleanos one-hot# Comparar con cada clase y asignar
        return convert(BitArray{2}, hcat([feature.== cl for cl in classes]...))   
    end
end


# Sobrecarga que extrae automáticamente las categorías y llama a la función principal
oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature))

# Sobrecarga para el caso booleano
oneHotEncoding(feature::AbstractArray{Bool,1}) = reshape(feature, length(feature), 1)



# Función para calcular los parámetros de normalización MinMax
function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real, 2})
  
    
    min_vals = minimum(dataset, dims=1)
    max_vals = maximum(dataset, dims=1)


    min_vals = reshape(min_vals, 1, length(min_vals))
    max_vals = reshape(max_vals, 1, length(max_vals))
  
    # Devolvemos como una tupla
    return (min_vals, max_vals)
end



# Función para calcular los parámetros de normalización Zero-Mean
function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real, 2})
    if isempty(dataset)
        error("El dataset está vacío.")
    end
    if any(isnan, dataset) || any(isinf, dataset)
        error("El dataset contiene valores NaN o Inf.")
    end

    means = mean(dataset, dims=1)
    std_devs = std(dataset, dims=1)

    # Evita divisiones por 0 reemplazando std_devs == 0 con 1
    std_devs = std_devs .+ (std_devs .== 0)

    means = reshape(means, 1, length(means))
    std_devs = reshape(std_devs, 1, length(std_devs))

    return (means, std_devs)
end


# Función para normalizar entre Min-Max, modificando la matriz original
function normalizeMinMax!(dataset::AbstractArray{<:Real, 2}, normalizationParameters::NTuple{2, AbstractArray{<:Real, 2}})
    min_vals, max_vals = normalizationParameters
    min_vals=Matrix(min_vals)
    max_vals=Matrix(max_vals)
    
    # Evitamos la división por cero en el caso de que el valor máximo sea igual al valor mínimo
    dataset .-= min_vals
    dataset .*= 1 ./ (max_vals .- min_vals) .^ (max_vals .!= min_vals)

end

# Función para normalizar entre Min-Max, calculando los parámetros antes de normalizar
function normalizeMinMax!(dataset::AbstractArray{<:Real, 2})
    # Calculamos los parámetros de normalización
    normalizationParameters = calculateMinMaxNormalizationParameters(dataset)

    # Normalizamos la matriz
    normalizeMinMax!(dataset, normalizationParameters)
end

# Función para normalizar entre Min-Max sin modificar la matriz original, usando parámetros previos
function normalizeMinMax(dataset::AbstractArray{<:Real, 2}, normalizationParameters::NTuple{2, AbstractArray{<:Real, 2}})
    
    # Hacemos una copia de la matriz para no modificar la original
    dataset_copy = copy(dataset)
    # Normalizamos la copia de la matriz
    normalizeMinMax!(dataset_copy, normalizationParameters)
    return dataset_copy
end

# Función para normalizar entre Min-Max sin modificar la matriz original, calculando parámetros antes de normalizar
function normalizeMinMax(dataset::AbstractArray{<:Real, 2})
    # Calculamos los parámetros de normalización
    normalizationParameters = calculateMinMaxNormalizationParameters(dataset)
    # Normalizamos y devolvemos la nueva matriz
    return normalizeMinMax(dataset, normalizationParameters)
end



function normalizeZeroMean!(dataset::AbstractArray{<:Real, 2}, normalizationParameters::NTuple{2, AbstractArray{<:Real, 2}})
    means, std_devs = normalizationParameters
    dataset .-= means
    dataset ./= std_devs
    dataset[:, vec(std_devs .== 0)] .= 0
    return dataset
end


function normalizeZeroMean!(dataset::AbstractArray{<:Real, 2})
    normalizationParameters = calculateZeroMeanNormalizationParameters(dataset)
    normalizeZeroMean!(dataset, normalizationParameters)
end

function normalizeZeroMean(dataset::AbstractArray{<:Real, 2}, normalizationParameters::NTuple{2, AbstractArray{<:Real, 2}})
    dataset_copy = copy(dataset)
    normalizeZeroMean!(dataset_copy, normalizationParameters)
    return dataset_copy
end


function normalizeZeroMean(dataset::AbstractArray{<:Real, 2})
    normalizationParameters = calculateZeroMeanNormalizationParameters(dataset)
    return normalizeZeroMean(dataset, normalizationParameters)
end

function classifyOutputs(outputs::AbstractArray{<:Real,1}; threshold::Real=0.5)
    return outputs.>=threshold
end;

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    if size(outputs, 2) == 1# dimensiones de la matriz solo una columna
        return reshape(classifyOutputs(outputs[:], threshold=threshold),:,1 ) #coge todas las filas y la primera columna 
    else 
        (_, indicesMaxEachInstance) = findmax(outputs, dims=2);
        classified = falses(size(outputs));
        classified[indicesMaxEachInstance] .= true
        return classified
    end
end;

function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    return mean((outputs .== targets))
end;

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    if size(targets, 2) == 1 # columnas = 2?  || size(outputs, 2) == 2
        return accuracy(vec(outputs), vec(targets)) #array multidimensional a columna 
    else
        classComparison = targets .== outputs
        correctClassifications = all(classComparison, dims=2)
        precision = mean(correctClassifications) 
        return precision
    end
end;

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
        return accuracy(outputs .>= threshold, targets)
end;

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)
    if size(targets, 2) == 1 
        return accuracy(vec(outputs), vec(targets))
    else
        classifiedOutputs = classifyOutputs(outputs)
        return accuracy(targets, classifiedOutputs)
    end
end;

function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
    # Inicializar la red neuronal vacía
    ann = Chain()

    # Variable para mantener el número de entradas de la capa actual
    numInputsLayer = numInputs

    # Construcción de las capas ocultas (si existen)
    if !isempty(topology)
        for (i, numOutputsLayer) in enumerate(topology)
            activation_function = transferFunctions[i]  # Selecciona la función de activación especificada
            ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, activation_function)) #añadimos capa oculta
            numInputsLayer = numOutputsLayer  # Actualiza el número de entradas para la próxima capa
        end
    end

    # Capa de salida según el número de clases
    if numOutputs == 1
        # Clasificación binaria (una sola salida con función sigmoide)
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, σ))
    else
        # Clasificación multiclase (uso de softmax para probabilidades)
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs), softmax)
    end

    return ann
    
end;



function trainClassANN(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; 
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    
    inputs, targets = dataset # separo la tupla de dos matrices que viene como parametro 

    # # Verificar que las entradas y las salidas no sean Nothing
    # if inputs == nothing || targets == nothing
    #    throw(ArgumentError("Las entradas o las salidas no pueden estar vacías."))
    # end

    # Asegurarse de que las entradas estén en Float32
    inputs = convert(Array{Float32}, inputs)

    numInputs = size(inputs, 2)   # Columnas de `inputs` = Número de características 
    numOutputs = size(targets, 2) # Columnas de `targets` = Número de clases 

    #Construcción de la RNA
    rna = buildClassANN(numInputs, topology, numOutputs, transferFunctions=transferFunctions)
    # Definir el optimizador
    opt_state = Flux.setup(Adam(learningRate), rna) 

    #Defino la funcion de perdidas
    loss(rna, x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(rna(x),y) : Losses.crossentropy(rna(x),y); #rna al principio no puede estar 
    
    # Inicializar el vector de pérdidas
    losses = Float32[]

    push!(losses,loss(rna,inputs',targets'))


    # Criterio de parada: entrenamiento hasta maxEpochs o minLoss alcanzado
    for epoch in 1:maxEpochs

        # Actualizar los pesos mediante backpropagation
        Flux.train!(loss, rna, [(inputs', targets')], opt_state)
        
        # Calcular el valor de la pérdida en el conjunto de entrenamiento
        currentLoss = loss(rna, inputs', targets')
        push!(losses, currentLoss)
        # Verificar si el criterio de parada ha sido alcanzado
        if currentLoss <= minLoss
            println("Criterio de parada alcanzado. Pérdida mínima alcanzada.")
            break
        end
        
        # Mostrar progreso cada ciertos ciclos
        if epoch % 100 == 0
            println("Epoch: $epoch, Loss: $currentLoss")
        end
    end
    
    return rna, losses
end


# Función para el caso de clasificación binaria
function trainClassANN(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; 
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)

    inputs, targets = dataset

    # Verificar que las entradas y las salidas no sean Nothing
    if inputs == nothing || targets == nothing
        throw(ArgumentError("Las entradas o las salidas no pueden ser Nothing."))
    end

    # Convertir las salidas (en caso de clasificación binaria) a una matriz de una columna
    targets = reshape(targets, :, 1)

    # Asegurar que las entradas sean de tipo Float32
    inputs = convert(Array{Float32}, inputs) #no deberia hacer falta 

    # Llamar a la función anterior para entrenar la RNA
    return trainClassANN(topology, (inputs, targets); transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate)
end

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 3 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random

function holdOut(N::Int, P::Real)
    @assert 0 ≤ P ≤ 1 "P debe estar entre 0 y 1"
    indices = randperm(N)  # Permutación aleatoria de los índices
    n_test = round(Int, P * N)  # Cantidad de patrones para test
    train_idx, test_idx = indices[1:end-n_test], indices[end-n_test+1:end]  # División en dos subconjuntos
    return train_idx, test_idx
end;

function holdOut(N::Int, Pval::Real, Ptest::Real)
    @assert Pval + Ptest <= 1 "La suma de Pval y Ptest debe ser menor o igual a 1"
    #Conjunto de test
    train_val_idx, test_idx = holdOut(N, Ptest)

    # Calcular la nueva tasa de validación relativa al conjunto de entrenamiento+validación
    new_Pval = Pval / (1 - Ptest)

    # Segunda separación: Conjunto de validación
    train_idx, val_idx = holdOut(length(train_val_idx), new_Pval)

    # Ajustar índices para que coincidan con el conjunto original
    train_idx = train_val_idx[train_idx]
    val_idx = train_val_idx[val_idx]

    return train_idx, val_idx, test_idx
end;

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0,size(trainingDataset[2],2))),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0,size(trainingDataset[2],2))),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)
    
    trainingInputs, trainingOutputs = trainingDataset
    validationInputs, validationOutputs = validationDataset
    testInputs, testOutputs = testDataset 

    trainingInputs =Float32.(trainingInputs)
    validationInputs= Float32.(validationInputs)
    testInputs=Float32.(testInputs)

    numInputs = size(trainingInputs,2)
    numOutputs = size(trainingOutputs,2)

    rna=buildClassANN(numInputs, topology, numOutputs, transferFunctions=transferFunctions)
    bestANN=deepcopy(rna) #almacenamos la mejor rna para el criterio de parada 

    # Definir el optimizador
    opt_state = Flux.setup(Adam(learningRate), rna)

    # Definir la función de pérdida
    loss(rna, x, y) = (size(y,1) == 1) ? Losses.binarycrossentropy(rna(x), y) : Losses.crossentropy(rna(x), y)

    trainLosses=Float32[]
    validLosses=Float32[]
    testLosses=Float32[]

    epochSinceBestANN=0


    #Loss inicial epoch = 0
    push!(trainLosses,loss(rna, trainingInputs',trainingOutputs'))
    
    # si existe conjunto de validacion, primer loss 
    if !isempty(validationDataset)
        validLoss = loss(rna, validationInputs', validationOutputs') 
        push!(validLosses, validLoss)
        bestValidLoss = validLosses[1]
    end

    if !isempty(testDataset)
        testLoss = loss(rna, testInputs', testOutputs')
        push!(testLosses, testLoss)
    end
    
    for epoch in 1:maxEpochs
        #backpropagation 
        Flux.train!(loss, rna, [(trainingInputs', trainingOutputs')], opt_state)

        currentLoss= loss(rna, trainingInputs', trainingOutputs')
        push!(trainLosses,currentLoss)
  
        if !isempty(validationDataset)
            validLoss= loss(rna,validationInputs', validationOutputs')
            push!(validLosses,validLoss)
        
            if validLoss < bestValidLoss
                bestANN =deepcopy(rna)
                bestValidLoss = validLoss
                epochSinceBestANN = 0
            else
                epochSinceBestANN +=1
            end
       
        end

        if !isempty(testDataset) #para no afectar al entreno, pero ver como evoluciona con cada ciclo
            testLoss = loss(rna, testInputs', testOutputs')
            push!(testLosses, testLoss)
        end 

        #si se ha pasado un conjunto validacion como parametro
        if !isempty(validationDataset)  
            print("Ciclo $epoch - Train loss: $currentLoss")
            print(validLoss !== nothing ? " - Validation Loss: $validLoss" : "")
            print(!isempty(testLosses) ? " Test loss : $(testLosses[end])" : "")  #ultimo valor loss de test 
            println()
        end;

        if currentLoss <= minLoss
            println("Criterio de parada alcanzado. Pérdida mínima alcanzada.")
            break
        end

        #Nuevo criterio parada, segun error de validacion 
        if !isempty(validationInputs) && epochSinceBestANN >= maxEpochsVal 
            println("Parada temprana ya que no hay mejoras en $maxEpochsVal épocas.")
            break
        end

    end

    # println("Train losses: ", trainLosses)
    # println("Validation losses: ", validLosses)
    # println("Test losses: ", testLosses)


     # Si hubo validación, devolvemos la mejor RNA, si no devolvemos la última entrenada
    return (!isempty(validationInputs) ? bestANN : rna), trainLosses, validLosses, testLosses

end

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0)),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)
    
    
    # Separar inputs y targets de cada dataset
    trainingInputs, trainingTargets = trainingDataset
    validationInputs, validationTargets = validationDataset
    testInputs, testTargets = testDataset

    # Convertir las salidas (targets) en matrices de una columna
    trainingTargets = reshape(trainingTargets, :, 1)
    validationTargets = reshape(validationTargets, :, 1)
    testTargets = reshape(testTargets, :, 1)

    # Llamar a la versión original de trainClassANN (con targets convertidos a matrices)
    return trainClassANN(topology, 
                        (trainingInputs, trainingTargets);
                        validationDataset=(validationInputs, validationTargets), 
                        testDataset=(testInputs, testTargets),
                        transferFunctions=transferFunctions, 
                        maxEpochs=maxEpochs, 
                        minLoss=minLoss, 
                        learningRate=learningRate, 
                        maxEpochsVal=maxEpochsVal)
end;




# ---------------------------------------------------------------------------------------------- 
# ------------------------------------- Ejercicio 4 --------------------------------------------
# ----------------------------------------------------------------------------------------------

function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    VN = sum(.!outputs .& .!targets)  # Verdaderos Negativos
    VP = sum(outputs .& targets)      # Verdaderos Positivos
    FP = sum(outputs .& .!targets)    # Falsos Positivos
    FN = sum(.!outputs .& targets)    # Falsos Negativos

    accuracy = (VP + VN) / (VP + VN + FP + FN)
    errorRate = 1 - accuracy

    sensitivity = (VP + FN == 0) ? 1.0 : VP / (VP + FN)
    specificity = (VN + FP == 0) ? 1.0 : VN / (VN + FP)
    precision = (VP + FP == 0) ? 1.0 : VP / (VP + FP)
    npv = (VN + FN == 0) ? 1.0 : VN / (VN + FN)
    F1 = (precision + sensitivity == 0) ? 0.0 : 2 * (precision * sensitivity) / (precision + sensitivity)

    return accuracy, errorRate, sensitivity, specificity, precision, npv, F1, [VN FP; FN VP]
end

function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    bin_outputs = outputs .>= threshold
    return confusionMatrix(bin_outputs, targets)
end

function printConfusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    # Llamamos a la función confusionMatrix para obtener los resultados
    accuracy, errorRate, sensitivity, specificity, precision, npv, F1, confMatrix = confusionMatrix(outputs, targets)
    
    # Mostramos los resultados por pantalla
    println("Confusion Matrix:")
    println(confMatrix)
    
    println("\nResultados:")
    println("Accuracy: ", accuracy)
    println("Error Rate: ", errorRate)
    println("Sensitivity: ", sensitivity)
    println("Specificity: ", specificity)
    println("Precision: ", precision)
    println("NPV: ", npv)
    println("F1 Score: ", F1)
end

function printConfusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    # Llamamos a la función confusionMatrix para convertir los outputs en valores binarios
    accuracy, errorRate, sensitivity, specificity, precision, npv, F1, confMatrix = confusionMatrix(outputs, targets; threshold=threshold)
    
    # Mostramos los resultados por pantalla
    println("Confusion Matrix:")
    println(confMatrix)
    
    println("\nResultados:")
    println("Accuracy: ", accuracy)
    println("Error Rate: ", errorRate)
    println("Sensitivity: ", sensitivity)
    println("Specificity: ", specificity)
    println("Precision: ", precision)
    println("NPV: ", npv)
    println("F1 Score: ", F1)
end


function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    n_classes = size(outputs, 2)

    # Inicialización de las métricas para cada clase
    sensitivities = zeros(n_classes)
    specificities = zeros(n_classes)
    precisions = zeros(n_classes)
    npvs = zeros(n_classes)
    F1s = zeros(n_classes)

    # Llamada a la función de la práctica anterior para cada clase
    for i in 1:n_classes
        tp = sum(outputs[:,i] .& targets[:,i])         # Verdaderos positivos
        tn = sum((.!outputs[:,i]) .& (.!targets[:,i])) # Verdaderos negativos
        fp = sum(outputs[:,i] .& (.!targets[:,i]))     # Falsos positivos
        fn = sum((.!outputs[:,i]) .& targets[:,i])     # Falsos negativos
        
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        npv = tn / (tn + fn)
        F1 = 2 * (precision * sensitivity) / (precision + sensitivity)
        
        # Asignación de métricas a las variables
        sensitivities[i] = sensitivity
        specificities[i] = specificity
        precisions[i] = precision
        npvs[i] = npv
        F1s[i] = F1
    end

    # Calcular la matriz de confusión
    confMatrix = [sum(outputs[:, i] .& targets[:, j]) for i in 1:n_classes, j in 1:n_classes]

    # Calcular métricas ponderadas o macro
    if weighted
        class_counts = vec(sum(targets, dims=1))  # Número de instancias por clase
        total = sum(class_counts)
        
        # Cálculo ponderado
        weighted_sensitivity = sum(sensitivities .* class_counts) / total
        weighted_specificity = sum(specificities .* class_counts) / total
        weighted_precision = sum(precisions .* class_counts) / total
        weighted_npvs = sum(npvs .* class_counts) / total
        weighted_F1 = sum(F1s .* class_counts) / total
        accuracy = weighted_sensitivity  # Usamos sensibilidad ponderada como precisión
    else
        accuracy = mean(sensitivities)
        weighted_sensitivity = mean(sensitivities)
        weighted_specificity = mean(specificities)
        weighted_precision = mean(precisions)
        weighted_npvs = mean(npvs)
        weighted_F1 = mean(F1s)
    end

    errorRate = 1 - accuracy

    return accuracy, errorRate, weighted_sensitivity, weighted_specificity, weighted_precision, weighted_npvs, weighted_F1, confMatrix
end

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true)
    # Convertir las salidas reales en booleanos usando el umbral
    outputs_bool = outputs .>= threshold
    # Llamar a la función principal de confusionMatrix para matrices booleanas
    return confusionMatrix(outputs_bool, targets; weighted=weighted)
end

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    # Verificar que todas las etiquetas estén en el vector de clases
    @assert all([in(label, classes) for label in vcat(targets, outputs)])
    
    # Convertir las salidas y objetivos a one-hot encoding
    outputs_encoded = oneHotEncoding(outputs, classes)
    targets_encoded = oneHotEncoding(targets, classes)
    
    # Llamar a la función principal de confusionMatrix para matrices booleanas
    return confusionMatrix(outputs_encoded, targets_encoded; weighted=weighted)
end

# Función para clasificación multiclase con clases calculadas automáticamente
function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    # Calcular las clases únicas a partir de las salidas y objetivos
    classes = unique(vcat(targets, outputs))
    # Llamar a la función anterior que requiere las clases como argumento
    return confusionMatrix(outputs, targets, classes; weighted=weighted)
end


using SymDoME

function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)

    # Convertir las entradas de entrenamiento a Float64
    trainingInputs = convert(Array{Float64}, trainingDataset[1])  # Entradas de entrenamiento (matriz)
    trainingTargets = trainingDataset[2]  # Etiquetas de entrenamiento (vector de booleanos, no se convierte)

    # Convertir las entradas de test a Float64
    testInputs = convert(Array{Float64}, testInputs)  # Entradas de test (matriz)

    # Llamar a la función dome para obtener el modelo
    _, _, _, model = dome(trainingInputs, trainingTargets; maximumNodes=maximumNodes)

    # Evaluar el modelo en el conjunto de test

    testOutputs = evaluateTree(model, testInputs)

    return testOutputs

    # Clasificar las salidas usando la función classifyOutputs
    #classifiedOutputs = classifyOutputs(testOutputs, threshold=0.0)
    #return classifiedOutputs (ASK!!!!!!!) 
end

function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    # Extraer las entradas y salidas del conjunto de entrenamiento
    trainingInputs = convert(Array{Float64}, trainingDataset[1])  # Entradas de entrenamiento (matriz)
    trainingTargets = trainingDataset[2] # Etiquetas entrenamiento amtriz booleana 
    
    numClasses = size(trainingTargets, 2) #numero de clases 
    
    #Caso clasificacion binaria
    if numClasses == 1
        trainingTargetsVector = vec(trainingTargets)
        
        binaryOutputs = trainClassDoME((trainingInputs, trainingTargetsVector), testInputs, maximumNodes)
        # Convertir las salidas a una matriz de una columna
        return reshape(binaryOutputs, :, 1)
    
    #if size(traingingDataset[2],2 ) > 2
    elseif numClasses == 2
        #Regla de uno contra uno
        numTestInstances = size(testInputs,2)
        outputs = zeros(Float64, numTestInstances, 2) #matriz que almacena las salidas
        
        
        binaryTargets_1 = vec(trainingTargets[:,1]) #usamos la primera columnna de las etiquetas binarias

        binaryOutputs_1 = trainClassDoME((trainingInputs, binaryTargets_1), testInputs, maximumNodes)

        # Almacenar las salidas en la primera columna
        outputs[:,1] = binaryOutputs_1

        binaryTargets_2 = trainingTargets[:,2]

        binaryOutputs_2 = trainClassDoME((trainingInputs, binaryTargets_2), testInputs, maximumNodes)
        
        # Almacenar salida en la segunda columna
        outputs[:,2] = binaryOutputs_2

        return outputs

    else 
        # Clasificación multiclase: aplicar la estrategia "uno contra todos"
        numTestInstances = size(testInputs, 1)  #columnas son las importante #era en filas ao final god damn!!!!!!!
        outputs = zeros(Float64, numTestInstances, numClasses)  # Matriz para almacenar las salidas
        for classIndex in 1:numClasses
            # etiquetas binarias de la clase actual
            binaryTargets = vec(trainingTargets[:, classIndex])

            # Llamar a la función trainClassDoME para clasificación binaria
            binaryOutputs = trainClassDoME((trainingInputs, binaryTargets), testInputs, maximumNodes)

            # Almacenar las salidas en la columna correspondiente
            outputs[:, classIndex] = binaryOutputs
        end

        return outputs

    end
end


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    trainingInputs, trainingTargets = trainingDataset

    trainingInputs = convert(Array{Float64}, trainingInputs)
    testInputs = convert(Array{Float64}, testInputs)

    classes = unique(trainingTargets)
    n_classes = length(classes)

    testOutputs = Array{eltype(trainingTargets),1}(undef, size(testInputs, 1))

    testOutputsDoME = trainClassDoME((trainingInputs, oneHotEncoding(trainingTargets, classes)), testInputs, maximumNodes)
    
    testOutputsBool = classifyOutputs(testOutputsDoME; threshold=0)

    if n_classes <=2
        testOutputsBool = vec(testOutputsBool)
        testOutputs[testOutputsBool] .= classes[1]
        if n_classes == 2
            testOutputs[.!testOutputsBool] .= classes[2]
        end
    else
        # Si es clasificación multiclase
        for i in 1:n_classes
            testOutputs[testOutputsBool[:, i]] .= classes[i]
        end
    end
    
    return testOutputs
end




# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 5 --------------------------------------------
# ----------------------------------------------------------------------------------------------



function crossvalidation(N::Int64, k::Int64)
    #
    # Codigo a desarrollar
    #
end;

function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    #
    # Codigo a desarrollar
    #
end;

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    #
    # Codigo a desarrollar
    #
end;

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    #
    # Codigo a desarrollar
    #
end;

function ANNCrossValidation(topology::AbstractArray{<:Int,1},
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
    crossValidationIndices::Array{Int64,1};
    numExecutions::Int=50,
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, validationRatio::Real=0, maxEpochsVal::Int=20)
    #
    # Codigo a desarrollar
    #
end;


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 6 --------------------------------------------
# ----------------------------------------------------------------------------------------------

#using MLJ
#using LIBSVM, MLJLIBSVMInterface
#using NearestNeighborModels, MLJDecisionTreeInterface

#SVMClassifier = MLJ.@load SVC pkg=LIBSVM verbosity=0
#kNNClassifier = MLJ.@load KNNClassifier pkg=NearestNeighborModels verbosity=0
#3DTClassifier  = MLJ.@load DecisionTreeClassifier pkg=DecisionTree verbosity=0


function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}, crossValidationIndices::Array{Int64,1})
   
end
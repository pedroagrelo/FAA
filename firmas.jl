
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

dataset = [1.0  5.0  3.0;
           2.0  6.0  3.0;
           3.0  7.0  3.0]  # La tercera columna es constante

means = mean(dataset, dims=1)
std_devs = std(dataset, dims=1)

normalizeZeroMean!(dataset, (means, std_devs))

println("Dataset normalizado:")
println(dataset)

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
    if size(targets, 2) == 1# dimensiones de la matriz solo una columna
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

    # Verificar que las entradas y las salidas no sean Nothing
    if inputs == nothing || targets == nothing
        throw(ArgumentError("Las entradas o las salidas no pueden estar vacías."))
    end

    # Asegurarse de que las entradas estén en Float32
    inputs = convert(Array{Float32}, inputs) 

    targets = convert(Array{Float32}, targets) #para comparar dos float 

    numInputs = size(inputs, 1)   # Filas de `inputs` = Número de características #antes estaba en columnas
    numOutputs = size(targets, 1) # Filas de `targets` = Número de clases # antes estaba en columnas

    #Construcción de la RNA
    rna = buildClassANN(numInputs, topology, numOutputs, transferFunctions=transferFunctions)

    # Definir el optimizador
    opt_state = Flux.setup(Adam(learningRate), rna) 

    #Defino la funcion de perdidas
    loss(x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(rna(x)',y') : Losses.crossentropy(rna(x)',y'); #rna al principio no puede estar #
    
    # Inicializar el vector de pérdidas
    losses = Float32[]

    # Criterio de parada: entrenamiento hasta maxEpochs o minLoss alcanzado
    for epoch in 1:maxEpochs
        
        # Calcular el valor de la pérdida en el conjunto de entrenamiento
        currentLoss = loss(inputs, targets) 
        push!(losses, currentLoss)
        
        # Verificar si el criterio de parada ha sido alcanzado
        if currentLoss <= minLoss
            println("Criterio de parada alcanzado. Pérdida mínima alcanzada.")
            break
        end

        # Actualizar los pesos mediante backpropagation
        Flux.train!(loss, rna, [(inputs', targets')], opt_state)

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
    inputs = convert(Array{Float32}, inputs)

    

    # Llamar a la función anterior para entrenar la RNA
    return trainClassANN(topology, (inputs, targets); transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate)
end


#using DelimitedFiles
#dataset = readdlm("iris.data",',');
#inputs = ann(inputs);
#inputs = dataset[:,1:4];


# Datos de ejemplo
X = rand(10, 100)  # 10 características, 100 patrones
Y = rand(Bool, 100)  # 100 salidas binarias (0 o 1)

# Convertir los datos a un formato adecuado
dataset = (X, Y)

# Definir la topología (2 capas ocultas con 5 y 3 neuronas)
topology = [5, 3]

# Entrenar la red
rna, losses = trainClassANN(topology, dataset)

# Imprimir las pérdidas de cada época
println(losses)


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 3 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random

function holdOut(N::Int, P::Real)
    #
    # Codigo a desarrollar
    #
end;

function holdOut(N::Int, Pval::Real, Ptest::Real)
    #
    # Codigo a desarrollar
    #
end;

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0,size(trainingDataset[2],2))),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0,size(trainingDataset[2],2))),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)
    #
    # Codigo a desarrollar
    #
end;

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0)),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)
    #
    # Codigo a desarrollar
    #
end;



# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 4 --------------------------------------------
# ----------------------------------------------------------------------------------------------


function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true)
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    #
    # Codigo a desarrollar
    #
end;

using SymDoME


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    #
    # Codigo a desarrollar
    #
end;

function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    #
    # Codigo a desarrollar
    #
end;


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    #
    # Codigo a desarrollar
    #
end;




# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 5 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random
using Random:seed!

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

using MLJ
using LIBSVM, MLJLIBSVMInterface
using NearestNeighborModels, MLJDecisionTreeInterface

SVMClassifier = MLJ.@load SVC pkg=LIBSVM verbosity=0
kNNClassifier = MLJ.@load KNNClassifier pkg=NearestNeighborModels verbosity=0
DTClassifier  = MLJ.@load DecisionTreeClassifier pkg=DecisionTree verbosity=0


function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}, crossValidationIndices::Array{Int64,1})
    #
    # Codigo a desarrollar
    #
end

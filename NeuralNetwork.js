// Weights and Biases Initialization:
function initializeNetwork(inputSize, hiddenSize, outputSize) {
    const weights = {
        inputToHidden: new Array(inputSize).fill(0).map(() => new Array(hiddenSize).fill(0).map(() => Math.random())), //array
        hiddenToOutput: new Array(hiddenSize).fill(0).map(() => new Array(outputSize).fill(0).map(() => Math.random()))
    };

    const biases = {
        hidden: new Array(hiddenSize).fill(0).map(() => Math.random()),
        output: new Array(outputSize).fill(0).map(() => Math.random())
    };

    return { weights, biases };
}

// Activaton Functions:
function sigmoid(x) { //expo -
    return 1 / (1 + Math.exp(-x));
}

function tanh(x) { 
    return Math.tanh(x);
}

function relu(x) {
    return Math.max(0, x);
}

function softmax(inputs) {  
    let maxInput = Math.max(...inputs);
    let expSum = inputs.reduce((sum, value) => sum + Math.exp(value - maxInput), 0);
    return inputs.map(value => Math.exp(value - maxInput) / expSum);
}

// Use created network biases and weights with actual input values to activate the initial linear algebra thesis:
function forwardPropagation(input, network, activationFunctionName) { //moves weights and biases between neurons

    let activationFunction;
  
    switch (activationFunctionName.toLowerCase()) {
        case 'sigmoid':
            activationFunction = sigmoid;
            break;
        case 'tanh':
            activationFunction = tanh;
            break;
        case 'relu':
            activationFunction = relu;
            break;
        case 'softmax':
            activationFunction = softmax;
            break;
        default:
            throw new Error('Invalid activation function name');
    }

    let hiddenInputs = new Array(network.weights.inputToHidden[0].length).fill(0);
    for (let i = 0; i < network.weights.inputToHidden.length; i++) {
        for (let j = 0; j < network.weights.inputToHidden[i].length; j++) {
            hiddenInputs[j] += input[i] * network.weights.inputToHidden[i][j];
            console.log(input[i]) 
        }
    }
  
    for (let i = 0; i < hiddenInputs.length; i++) {
        hiddenInputs[i] += network.biases.hidden[i];
    }

    let hiddenOutputs = hiddenInputs.map(activationFunction);

    network.hiddenLayerOutput = hiddenOutputs;

    let outputInputs = new Array(network.weights.hiddenToOutput[0].length).fill(0);
    for (let i = 0; i < network.weights.hiddenToOutput.length; i++) {
        for (let j = 0; j < network.weights.hiddenToOutput[i].length; j++) {
            outputInputs[j] += hiddenOutputs[i] * network.weights.hiddenToOutput[i][j];
           
            
        }
    }

    for (let i = 0; i < outputInputs.length; i++) {
    
        outputInputs[i] += network.biases.output[i];
   
    }

    let finalOutput = outputInputs.map(activationFunction);

    return finalOutput;
}

// Derivative of activation functions for Backward Propagation: to fix learning rate
function sigmoidDerivative(output) {
    return output * (1 - output);
}

function tanhDerivative(output) {
    return 1 - Math.pow(Math.tanh(output), 2);
}

function reluDerivative(output) {
    return output > 0 ? 1 : 0;
}

function softmaxDerivative(outputSoftmax, target) {
    return outputSoftmax.map((output, i) => output - target[i]);
}


// Create a Backward Propagation to calculate error rate and accuracy and relevant values.
function backPropagation(network, output, expectedOutput, activationFunctionName, learningRate,input) {
    let activationDerivative;
   
    switch (activationFunctionName) {
        case 'sigmoid':
            activationDerivative = sigmoidDerivative;
            break;
        case 'tanh':
            activationDerivative = tanhDerivative;
            break;
        case 'relu':
            activationDerivative = reluDerivative;
            break;
        case 'softmax':
            activationDerivative = softmaxDerivative;
            break;
        default:
            throw new Error('Invalid activation function name');
    }

    const outputErrors = output.map((o, i) => expectedOutput[i] - o);
 
    const outputGradients = outputErrors.map((error, i) => error * activationDerivative(output[i]));

    for (let i = 0; i < network.weights.hiddenToOutput.length; i++) {
        for (let j = 0; j < network.weights.hiddenToOutput[i].length; j++) {
            network.weights.hiddenToOutput[i][j] += learningRate * outputGradients[j] * network.hiddenLayerOutput[i];
            
        }
      
        network.biases.output[i] += learningRate * outputGradients[i];
    }

    let hiddenErrors = new Array(network.weights.inputToHidden[0].length).fill(0);
    for (let i = 0; i < network.weights.hiddenToOutput.length; i++) {
        for (let j = 0; j < network.weights.hiddenToOutput[i].length; j++) {
            hiddenErrors[i] += outputGradients[j] * network.weights.hiddenToOutput[i][j];
        }
    }

    const hiddenGradients = hiddenErrors.map((error, i) => error * activationDerivative(network.hiddenLayerOutput[i]));

    for (let i = 0; i < network.weights.inputToHidden.length; i++) {
        for (let j = 0; j < network.weights.inputToHidden[i].length; j++) {
            network.weights.inputToHidden[i][j] += learningRate * hiddenGradients[j] * input[i];
        }
        network.biases.hidden[i] += learningRate * hiddenGradients[i];
    }
}


// Use Forward & Backward Propagationeturn and then return the outputs of the neural network
async function trainNetwork(data, network, epochs, learningRate, activationFunctionName, uniqueClasses) {
    let input;
    let classLabel;
    let expectedOutput;
    
    for (let epoch = 0; epoch < epochs; epoch++) {
        
        for (const row of data) {
          
            input= row.slice(0, 1);
          
            classLabel = row.slice(2,5);
          
            expectedOutput = convertClassToOutput(classLabel, uniqueClasses);
         
         
            backPropagation(network,output, expectedOutput,activationFunctionName, learningRate,input);
        }
        let output =forwardPropagation(input, network, activationFunctionName);
        
        await new Promise(resolve => setTimeout(resolve, 0));
    }

    // Return the trained network
    return network;
}


function convertClassToOutput(classLabel, uniqueClasses) { //each one is a column, ex: 0,0,1 1 apple index2
    let output = new Array(uniqueClasses.length).fill(0);
    const index = uniqueClasses.indexOf(classLabel);
    if (index !== -1) {
        output[index] = 1;
    }

    return output;
}

function predict(network, inputData, activationFunctionName) { //user puts input in testing page and linear algebra expects the result
    const input = preprocessInput(inputData.image, inputData.jsonData);
    const output = forwardPropagation(input, network, activationFunctionName);
    const predictedClassIndex = output.indexOf(Math.max(...output));
    return predictedClassIndex;
}

export { initializeNetwork, trainNetwork, predict }
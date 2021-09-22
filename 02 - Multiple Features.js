const DERIVATIVE_STEP = 0.0001;

const calculateNeuron = (inputs, weights, bias) => bias + inputs.reduce((sum, input, index) => sum + input * weights[index], 0);

const calculateLoss = (y, y1) => (y - y1) ** 2;

const getBatch = (input, batchSize) => {
    const arrOfIndexes = Array.from({ length: batchSize }, (v, i) => i);
    const randomIndexes = [];
    for (let i = 0; i < batchSize; i += 1) {
        const randomIndex = Math.floor(Math.random() * arrOfIndexes.length);
        randomIndexes.push(arrOfIndexes[randomIndex]);
        arrOfIndexes.splice(randomIndex, 1);
    }
    return randomIndexes.map(index => input[index]);
};

const calculateAverageGradient = (data, weights, bias, batchSize, parameterWithStep) => {
    // get a random batch of examples from the whole data set
    const batch = getBatch(data, batchSize)

    // calculate a gradient for each example and return the average
    const gradients = batch.map((example) => {
        // approximate the derivative: (loss(a + h) - loss(a)) / h
        const loss = calculateLoss(example.label, calculateNeuron(example.features, weights, bias));
        const withStep = (parameter, parameterName) => {
            if (parameterWithStep === 'bias' && parameterName === 'bias') return parameter + DERIVATIVE_STEP
            else if (typeof parameterWithStep === 'number' && parameterName === 'weight') {
                const modifiedWeights = [...parameter]
                modifiedWeights[parameterWithStep] += DERIVATIVE_STEP
                return modifiedWeights
            }
            else return parameter
        }
        const loss2 = calculateLoss(example.label, calculateNeuron(example.features, withStep(weights, 'weight'), withStep(bias, 'bias')));
        return -1 * ((loss2 - loss) / DERIVATIVE_STEP);
    });
    return gradients.reduce((sum, gradient) => sum + gradient, 0) / gradients.length;
}

const gradientDescent = (data, options) => {
    const { epochs, learningRate, batchSize } = options;

    // initialize with starter values
    let weights = Array(data[0].features.length).fill().map(Math.random);
    let bias = 0;

    // update values with gradient descent
    for (let epoch = 0; epoch < epochs; epoch += 1) {
        bias += calculateAverageGradient(data, weights, bias, batchSize, 'bias') * learningRate
        weights.forEach((weight, index) => {
            weights[index] += calculateAverageGradient(data, weights, bias, batchSize, index) * learningRate
        });

        const RMSE = Math.sqrt(data.reduce((sum, example) => sum + calculateLoss(example.label, calculateNeuron(example.features, weights, bias)), 0));
        console.log(`Epoch ${epoch} - weights: [${weights}], bias: ${bias}, RMSE: ${RMSE}`)
    }
    return { weights, bias }
}

// generate random data with multiple features, similar to googles crash course dataset
const generateData = (size, noise) => {
    return Array(size).fill(null).map(() => {
        // set the label randomly to either -1 or 1
        const label = Math.sign(Math.random() - .5)

        // if the label is -1 points will be centered around -2, -2; if its 1 it'll be 2, 2
        const x = 2 * label + (Math.random() - .5) * noise
        const y = 2 * label + (Math.random() - .5) * noise
        return { features: [x, y], label }
    });
}

const data = generateData(100, 20)
const { weights, bias } = gradientDescent(data, {epochs: 1000, learningRate: 0.0001, batchSize: 10})

// check if the learned values are correct
data.slice(0, 5).forEach(({ features, label }) => {
    console.log(features, label, Math.sign(calculateNeuron(features, weights, bias)))
})

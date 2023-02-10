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

const crossFeatures = (...features) => features.reduce((result, feature) => result * feature, 1);

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

// generate random data with multiple features, in a checkerboard pattern
const generateData = (size, noise) => {
    return Array(size).fill(null).map(() => {
        // generate a random center point in a random quadrant and assign a label
        const center = Array(2).fill(null).map(() => Math.random() > .5 ? 1 : -1)
        const label = center[0] * center[1]

        // generate a random point around the center point
        const x = 2 * center[0] + (Math.random() - .5) * noise
        const y = 2 * center[1] + (Math.random() - .5) * noise
        return { features: [x, y], label }
    });
}
const data = generateData(100, 3)

// add a feature cross to the data
const dataWithFeatureCross = data.map(({ features, label }) => {
    const [x, y] = features
    return { features: [x, y, crossFeatures(x, y)], label }
})
// run gradient descent and check if the learned values are correct
const { weights, bias } = gradientDescent(dataWithFeatureCross, {epochs: 1000, learningRate: 0.0005, batchSize: 20})
dataWithFeatureCross.slice(0, 5).forEach(({ features, label }) => {
    const result = calculateNeuron(features, weights, bias)
    const coordinates = features.map(x => x.toFixed(3))
    const correct = Math.sign(result) === label
    console.log(`Input coordinates: [${coordinates}], Expected: ${label} Result: ${result.toFixed(3)}, Correct: ${correct}`)
})

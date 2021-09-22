const DERIVATIVE_STEP = 0.0001;

const calculateNeuron = (input, weight, bias) => input * weight + bias;

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

const calculateAverageGradient = (data, weight, bias, batchSize, parameterWithStep) => {
    // get a random batch of examples from the whole data set
    const batch = getBatch(data, batchSize)

    // calculate a gradient for each example and return the average
    const gradients = batch.map((example) => {
        // approximate the derivative: (loss(a + h) - loss(a)) / h
        const loss = calculateLoss(example.label, calculateNeuron(example.feature, weight, bias));
        const withStep = (parameter, parameterName) => parameter + (parameterWithStep === parameterName ? DERIVATIVE_STEP : 0)
        const loss2 = calculateLoss(example.label, calculateNeuron(example.feature, withStep(weight, 'weight'), withStep(bias, 'bias')));
        return -1 * ((loss2 - loss) / DERIVATIVE_STEP);
    });
    return gradients.reduce((sum, gradient) => sum + gradient, 0) / gradients.length;
}

const gradientDescent = (data, options) => {
    const { epochs, learningRate, batchSize } = options;

    // initialize with starter values
    let weight = Math.random();
    let bias = 0;

    // update values with gradient descent
    for (let epoch = 0; epoch < epochs; epoch += 1) {
        bias += calculateAverageGradient(data, weight, bias, batchSize, 'bias') * learningRate
        weight = weight + calculateAverageGradient(data, weight, bias, batchSize, 'weight') * learningRate

        const RMSE = Math.sqrt(data.reduce((sum, example) => sum + calculateLoss(example.label, calculateNeuron(example.feature, weight, bias)), 0));
        console.log(`Epoch ${epoch} - weight: ${weight}, bias: ${bias}, RMSE: ${RMSE}`)
    }
    return { weight, bias }
}

// generate random data
const data = Array(100).fill(null).map(() => {
    const random = Number(Math.random().toFixed(2));
    return {label: random / 3 + 2, feature: random};
});

gradientDescent(data, {epochs: 1000, learningRate: 0.1, batchSize: 10})


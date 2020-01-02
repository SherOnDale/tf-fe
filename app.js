class LinearRegression {
  constructor(features, labels, options) {
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);
    this.mseHistory = [];

    this.options = Object.assign(
      {
        learningRate: 0.1,
        iterations: 1000,
        batchSize: 10
      },
      options
    );

    this.weights = tf.zeros([this.features.shape[1], 1]);

    this.m = 0;
    this.b = 0;
  }

  gradientDescent(features, labels) {
    const currentGuesses = features.matMul(this.weights);
    const differences = currentGuesses.sub(labels);

    const gradients = features
      .transpose()
      .matMul(differences)
      .div(features.shape[0]);

    this.weights = this.weights.sub(gradients.mul(this.options.learningRate));
  }

  train() {
    const batchQuantity = Math.floor(
      this.features.shape[0] / this.options.batchSize
    );
    for (let i = 0; i < this.options.iterations; i++) {
      for (let j = 0; j < batchQuantity; j++) {
        const startIndex = j * this.options.batchSize;
        const { batchSize } = this.options;
        const featureSlice = this.features.slice(
          [startIndex, 0],
          [batchSize, -1]
        );
        const labelSize = this.labels.slice([startIndex, 0], [batchSize, -1]);
        this.gradientDescent(featureSlice, labelSize);
      }
      this.recordMSE();
      this.updateLearningRate();
    }
  }

  predict(observations) {
    return this.processFeatures(observations).matMul(this.weights);
  }

  test(testFeatures, testLabels) {
    testFeatures = this.processFeatures(testFeatures);
    testLabels = tf.tensor(testLabels);

    const predictions = testFeatures.matMul(this.weights);

    const res = testLabels
      .sub(predictions)
      .pow(2)
      .sum()
      .dataSync();

    const tot = testLabels
      .sub(testLabels.mean())
      .pow(2)
      .sum()
      .dataSync();

    return 1 - res / tot;
  }

  processFeatures(features) {
    features = tf.tensor(features);

    if (this.mean && this.variance) {
      features = features.sub(this.mean).div(this.variance.pow(0.5));
    } else {
      features = this.standardize(features);
    }

    features = tf.ones([features.shape[0], 1]).concat(features, 1);

    return features;
  }

  standardize(features) {
    const { mean, variance } = tf.moments(features, 0);

    this.mean = mean;
    this.variance = variance;

    return features.sub(mean).div(variance.pow(0.5));
  }

  recordMSE() {
    const mse = this.features
      .matMul(this.weights)
      .sub(this.labels)
      .pow(2)
      .sum()
      .div(this.features.shape[0])
      .dataSync();
    this.mseHistory.unshift(mse);
  }

  updateLearningRate() {
    if (this.mseHistory.length < 2) {
      return;
    }

    if (this.mseHistory[0] > this.mseHistory[1]) {
      this.options.learningRate /= 2;
    } else {
      this.options.learningRate *= 1.05;
    }
  }
}

let currentModel;
let apiData;
const msg = document.querySelector('#message');
const url = document.querySelector('#url');

document.querySelector('#train-btn').addEventListener('click', () => {
  if (url.value) {
    fetch(url.value)
      .then(res => res.json())
      .then(data => {
        apiData = data;
        const { features, labels } = apiData;
        currentModel = new LinearRegression(features, labels, {
          learningRate: 0.1,
          iterations: 3,
          batchSize: 10
        });
        currentModel.train();
        msg.innerHTML = 'Model trained';
      });
  } else {
    msg.innerHTML = 'Enter the data URL';
  }
});

document.querySelector('#test-btn').addEventListener('click', () => {
  if (currentModel) {
    const { testFeatures, testLabels } = apiData;
    msg.innerHTML = `The Coefficient of Determination is ${currentModel.test(
      testFeatures,
      testLabels
    )}`;
  } else {
    msg.innerHTML = 'Model not trained yet';
  }
});

document.querySelector('#predict-btn').addEventListener('click', () => {
  if (currentModel) {
    msg.innerHTML = currentModel.predict([[120, 380, 2]]).dataSync();
  } else {
    msg.innerHTML = 'Model not trained yet';
  }
});

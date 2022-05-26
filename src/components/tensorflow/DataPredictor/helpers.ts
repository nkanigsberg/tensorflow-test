import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import type { Car } from "./DataPredictor";

type CarSource = {
  Miles_per_Gallon: number;
  Horsepower: number;
};

export type Options = {
  learningRate?: number;
  epochs?: number;
  onEpochEnd?: (epoch: number, logs?: any) => Promise<void>;
  plotMin?: number;
  plotMax?: number;
};

export const getData = async () => {
  const carsDataResponse = await fetch(
    "https://storage.googleapis.com/tfjs-tutorials/carsData.json"
  );
  const carsData: CarSource[] = await carsDataResponse.json();
  // console.log(carsData);
  const cleaned: Car[] = carsData
    .map((car) => ({
      mpg: car.Miles_per_Gallon,
      horsepower: car.Horsepower,
    }))
    .filter((car) => car.mpg != null && car.horsepower != null);
  return cleaned;
};

/**
 * Convert the input data to tensors that we can use for machine
 * learning. We will also do the important best practices of _shuffling_
 * the data and _normalizing_ the data
 * MPG on the y-axis.
 */
export function convertToTensor(data: Car[]) {
  // Wrapping these calculations in a tidy will dispose any
  // intermediate tensors.

  return tf.tidy(() => {
    // Step 1. Shuffle the data
    tf.util.shuffle(data);

    // Step 2. Convert data to Tensor
    const inputs = data.map((d) => d.horsepower);
    const labels = data.map((d) => d.mpg);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();
    // console.log(inputMax.dataSync());
    // console.log(inputMin.dataSync());
    // console.log(labelMax.dataSync());
    // console.log(labelMin.dataSync());

    const normalizedInputs = inputTensor
      .sub(inputMin)
      .div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor
      .sub(labelMin)
      .div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      // Return the min/max bounds so we can use them later.
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    };
  });
}

export const createModel = () => {
  // Define a model for linear regression.
  const model = tf.sequential();

  // Add a single input layer
  model.add(tf.layers.dense({ inputShape: [1], units: 32 }));

  // Add a hidden layer
  model.add(tf.layers.dense({ units: 32, activation: "sigmoid" }));

  // Add an output layer
  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

  return model;
};

export async function trainModel(
  model: tf.Sequential,
  inputs: tf.Tensor<tf.Rank>,
  labels: tf.Tensor<tf.Rank>,
  options?: Options
) {
  // Prepare the model for training.
  model.compile({
    optimizer: tf.train.adam(options?.learningRate),
    loss: tf.losses.meanSquaredError,
    metrics: ["mse"],
  });

  const batchSize = 32;

  return await model.fit(inputs, labels, {
    batchSize,
    epochs: options?.epochs ?? 50,
    shuffle: true,
    callbacks: [
      { onEpochEnd: options?.onEpochEnd },
      tfvis.show.fitCallbacks(
        { name: "Training Performance" },
        ["loss", "mse"],
        { height: 200, callbacks: ["onEpochEnd"] }
      ),
    ],
  });
}

export function testModel(
  model: tf.Sequential,
  inputData: Car[],
  normalizationData: Record<string, tf.Tensor<tf.Rank>>,
  plotMin?: number,
  plotMax?: number
) {
  const { inputMax, inputMin, labelMin, labelMax } = normalizationData;

  // Generate predictions for a uniform range of numbers between 0 and 1;
  // We un-normalize the data by doing the inverse of the min-max scaling
  // that we did earlier.
  const [xs, preds] = tf.tidy(() => {
    const min = inputMin.dataSync()[0];
    const max = inputMax.dataSync()[0];
    const difference = max - min;
    // scale the test data to correspond to user specified range
    const xmin = plotMin === undefined ? 0 : 0 - (min - plotMin) / difference;
    const xmax = plotMax === undefined ? 1 : 1 + (plotMax - max) / difference;
    const xs = tf.linspace(xmin, xmax, 100);

    const preds = model.predict(xs.reshape([100, 1]));

    const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin);
    // const unNormXs = xs
    //   .mul(tf.tensor1d([300]).sub(tf.tensor1d([0])))
    //   .add(tf.tensor1d([0]));

    const unNormPreds = (preds as tf.Tensor<tf.Rank>)
      .mul(labelMax.sub(labelMin))
      .add(labelMin);

    // Un-normalize the data
    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });

  const predictedPoints = Array.from(xs).map((val, i) => {
    return { x: val, y: preds[i] };
  });

  const originalPoints = inputData.map((d) => ({
    x: d.horsepower,
    y: d.mpg,
  }));

  tfvis.render.scatterplot(
    { name: "Model Predictions vs Original Data" },
    {
      values: [originalPoints, predictedPoints],
      series: ["original", "predicted"],
    },
    {
      xLabel: "Horsepower",
      yLabel: "MPG",
      height: 300,
    }
  );

  return predictedPoints;
}

export const predict = async (
  data: Car[],
  model: tf.Sequential,
  options?: Options
) => {
  tfvis.show.modelSummary({ name: "Model Summary" }, model);

  const values = data.map((d) => ({
    x: d.horsepower,
    y: d.mpg,
  }));

  // display ts scatterplot
  tfvis.render.scatterplot(
    { name: "Horsepower v MPG" },
    { values },
    {
      xLabel: "Horsepower",
      yLabel: "MPG",
      height: 300,
    }
  );

  // Convert the data to a form we can use for training.
  const tensorData = convertToTensor(data);
  const { inputs, labels } = tensorData;

  // Train the model
  await trainModel(model, inputs, labels, options);
  console.log("Done Training");

  // Make some predictions using the model and compare them to the
  // original data
  return testModel(model, data, tensorData, options?.plotMin, options?.plotMax);
};

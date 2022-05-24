import { useState } from "react";
import * as tf from "@tensorflow/tfjs";
import { Chart, registerables } from "chart.js";
import { Chart as ReactChart, Line } from "react-chartjs-2";

import styles from "./classifier.module.css";

Chart.register(...registerables);

function Classifier() {
  const [xValues, setXValues] = useState<number[]>([1, 2, 3, 4]);
  const [yValues, setYValues] = useState<number[]>([1, 3, 5, 7]);
  const [forecastRange, setForecastRange] = useState<number>(10);
  const [predictX, setPredictX] = useState<number[]>(
    Array.from(Array(forecastRange).keys())
  );
  const [predictY, setPredictY] = useState<Float32Array>();
  const [numIterations, setNumIterations] = useState<number>(100);

  const handleClick = async () => {
    // Define a model for linear regression.
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

    model.compile({ loss: "meanSquaredError", optimizer: "sgd" });

    // Generate some synthetic data for training.
    const xs = tf.tensor2d(xValues, [4, 1]);
    const ys = tf.tensor2d(yValues, [4, 1]);

    // console.log("xs", (await xs.buffer()).values);
    // console.log("ys", (await ys.buffer()).values);

    // Train the model using the data.
    model.fit(xs, ys, { epochs: numIterations }).then(() => {
      // console.log(
      //   (
      //     model.predict(
      //       tf.tensor2d([2, 4, 6, 8], [predictX.length, 1])
      //     ) as tf.Tensor
      //   ).dataSync()
      // );
      // Use the model to do inference on a data point the model hasn't seen before:
      setPredictY(
        (
          model.predict(
            tf.tensor2d(predictX, [predictX.length, 1])
          ) as tf.Tensor
        ).dataSync() as Float32Array
      );
      // Open the browser devtools to see the output
    });
  };

  return (
    <div>
      <h2>Linear Prediction</h2>
      {/* <p>
        Original example taken from:
        <a href="https://www.tensorflow.org/js/tutorials/setup#expandable-2">
          https://www.tensorflow.org/js/tutorials/setup#expandable-2
        </a>
      </p> */}
      <label>
        # of iterations:
        <input
          type="number"
          onChange={(e) => setNumIterations(parseInt(e.target.value))}
          value={numIterations}
        />
      </label>
      <button type="button" onClick={handleClick}>
        Predict
      </button>
      {/* {predictY && <p>Predicted y value: {predictY.toString()}</p>} */}

      <div className={styles.chartContainer}>
        <ReactChart
          className={styles.chart}
          type="scatter"
          data={{
            datasets: [
              {
                type: "scatter",
                label: "Training set",
                data: xValues.map((x, i) => ({
                  x: x,
                  y: yValues[i],
                })),
                backgroundColor: "rgb(255, 99, 132)",
              },
              {
                type: "line",
                label: "Prediction",
                data: Array.from(Array(forecastRange).keys()).map((x, i) => ({
                  x: x,
                  y: predictY?.[i],
                })),
                backgroundColor: "rgb(54, 162, 235)",
              },
            ],
            labels: Array.from(Array(forecastRange).keys()),
          }}
        />
      </div>
    </div>
  );
}

export default Classifier;

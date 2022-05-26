import { useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import { Chart, registerables } from "chart.js";
import { Chart as ReactChart } from "react-chartjs-2";

import {
  getData,
  predict,
  convertToTensor,
  testModel,
  createModel,
  generateSinePoints,
} from "./helpers";

import styles from "./DataPredictor.module.css";

export type Car = {
  mpg: number;
  horsepower: number;
};

Chart.register(...registerables);

function DataPredictor() {
  const [data, setData] = useState<Car[]>([]);
  const [predictions, setPredictions] = useState<{ x: number; y: number }[]>(
    []
  );
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [loss, setLoss] = useState<number>();
  const [epochs, setEpochs] = useState<number>(20);
  const [learningRate, setLearningRate] = useState<number>(0.01);
  const [plotMin, setPlotMin] = useState<number>();
  const [plotMax, setPlotMax] = useState<number>();
  const [dataset, setDataset] = useState<string>("cars");

  let model: tf.Sequential;

  // fetch data on mount
  useEffect(() => {
    (async () => {
      const data =
        dataset === "cars"
          ? await getData()
          : generateSinePoints(5).map(({ x, y }) => ({
              horsepower: x,
              mpg: y,
            }));
      setData(data);
    })();
  }, [dataset]);

  /** Custom callback to run at the end of each epoch */
  const onEpochEnd = async (epoch: number, logs?: tf.Logs) => {
    setCurrentEpoch(epoch);
    setLoss(logs?.loss);
    setPredictions(
      testModel(model, data, convertToTensor(data), plotMin, plotMax)
    );
  };

  const handlePredict = async () => {
    model = createModel();

    await predict(data, model, {
      epochs,
      onEpochEnd,
      learningRate: learningRate,
      plotMin,
      plotMax,
    });
    // console.log(predictedPoints);
    // setPredictions(predictedPoints);
  };

  return (
    <div>
      <h2>2D data prediction</h2>
      <fieldset>
        <legend>Dataset</legend>
        <label>
          Cars
          <input
            type="radio"
            name="predictionType"
            onChange={(e) => setDataset(e.target.value)}
            value="cars"
            defaultChecked
          />
        </label>
        <label>
          Sine
          <input
            type="radio"
            name="predictionType"
            onChange={(e) => setDataset(e.target.value)}
            value="sine"
          />
        </label>
      </fieldset>
      <button
        type="button"
        onClick={() => tfvis.visor().open()}
        className={styles.visorButton}
      >
        Show visor
      </button>
      <label>
        Learning rate:
        <input
          type="number"
          onChange={(e) => setLearningRate(parseFloat(e.target.value))}
          value={learningRate}
        />
      </label>
      <label>
        # of epochs:
        <input
          type="number"
          onChange={(e) => setEpochs(parseInt(e.target.value))}
          value={epochs}
        />
      </label>
      <label>
        Plot min:
        <input
          type="number"
          onChange={(e) => setPlotMin(parseInt(e.target.value))}
          value={plotMin}
        />
      </label>
      <label>
        Plot max:
        <input
          type="number"
          onChange={(e) => setPlotMax(parseInt(e.target.value))}
          value={plotMax}
        />
      </label>
      <button type="button" onClick={handlePredict}>
        Predict
      </button>

      {/* <div> */}
      <div>Current epoch: {currentEpoch}</div>
      {loss && <div>Loss: {loss.toFixed(8)}</div>}
      {/* </div> */}

      <div className={styles.chartContainer}>
        <ReactChart
          className={styles.chart}
          type="scatter"
          data={{
            datasets: [
              {
                type: "scatter",
                label: "Training set",
                data: data.map((car) => ({
                  x: car.horsepower,
                  y: car.mpg,
                })),
                backgroundColor: "rgb(255, 99, 132)",
              },
              {
                type: "line",
                label: "Prediction",
                data: predictions,
                backgroundColor: "rgb(75, 192, 192)",
                borderColor: "rgb(75, 192, 192)",
                pointRadius: 0,
              },
            ],
          }}
          options={{
            scales: {
              x: {
                title: {
                  display: true,
                  text: "Horsepower",
                },
                // suggestedMin: 0,
              },
              y: {
                title: {
                  display: true,
                  text: "Miles per gallon",
                },
              },
            },
            animation: {
              duration: 0,
            },
          }}
        />
      </div>
    </div>
  );
}

export default DataPredictor;

import { useEffect, useState } from "react";
import * as tfvis from "@tensorflow/tfjs-vis";
import { Chart, registerables } from "chart.js";
import { Chart as ReactChart } from "react-chartjs-2";

import { getData, predict } from "./helpers";

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
  const [epochs, setEpochs] = useState<number>(50);

  // fetch data on mount
  useEffect(() => {
    (async () => {
      const data = await getData();
      setData(data);
    })();
  }, []);

  const handlePredict = async () => {
    const [originalPoints, predictedPoints] = await predict(data, { epochs });
    console.log(originalPoints, predictedPoints);
    setPredictions(predictedPoints);
  };

  return (
    <div>
      <h2>2D data prediction</h2>
      {/* <fieldset>
        <legend>Prediction type</legend>
        <label>
          Linear
          <input type="radio" name="predictionType" />
        </label>
        <label>
          Non-linear
          <input type="radio" name="predictionType" />
        </label>
      </fieldset> */}
      <button
        type="button"
        onClick={() => tfvis.visor().open()}
        className={styles.visorButton}
      >
        Show visor
      </button>
      <label>
        # of epochs:
        <input
          type="number"
          onChange={(e) => setEpochs(parseInt(e.target.value))}
          value={epochs}
        />
      </label>
      <button type="button" onClick={handlePredict}>
        Predict
      </button>

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

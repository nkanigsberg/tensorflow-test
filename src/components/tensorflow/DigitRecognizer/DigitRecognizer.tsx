import { useRef, useState } from "react";

import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";

import { DigitCanvas } from "./DigitCanvas";

import {
  showExamples,
  getModel,
  train,
  showAccuracy,
  showConfusion,
} from "./helpers";

import { MnistData } from "./data";

import styles from "./DigitRecognizer.module.css";

const CANVAS_SCALE = 5;

function DigitRecognizer() {
  const [prediction, setPrediction] = useState<string>("");
  const [model, setModel] = useState<tf.Sequential>();

  const canvasRef = useRef<HTMLCanvasElement>(null);

  const run = async () => {
    const data = new MnistData();
    await data.load();
    await showExamples(data);
    const model = getModel();
    setModel(getModel());

    tfvis.show.modelSummary(
      { name: "Model Architecture", tab: "Model" },
      model
    );

    await train(model, data);

    await showAccuracy(model, data);
    await showConfusion(model, data);
  };

  const recognize = () => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");

    if (model && canvas && ctx) {
      const imageData = ctx.getImageData(0, 0, 28, 28);

      const IMAGE_SIZE = 784;

      const datasetBytesBuffer = new ArrayBuffer(IMAGE_SIZE * 4);

      const datasetBytesView = new Float32Array(
        datasetBytesBuffer,
        0,
        IMAGE_SIZE
      );

      for (let i = 0; i < imageData.data.length / 4; i++) {
        // All channels hold an equal value since the image is grayscale, so
        // just read the red channel.
        datasetBytesView[i] = imageData.data[i * 4] / 255;
      }
      const datasetImages = new Float32Array(datasetBytesBuffer);
      const batchImagesArray = new Float32Array(IMAGE_SIZE);

      batchImagesArray.set(datasetImages);

      const xs = tf.tensor2d(batchImagesArray, [1, IMAGE_SIZE]);

      const IMAGE_WIDTH = 28;
      const IMAGE_HEIGHT = 28;

      const testxs = xs.reshape([1, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);

      const preds: tf.Tensor<tf.Rank.R1> = (
        model.predict(testxs) as tf.Tensor1D
      ).argMax(-1);

      testxs.dispose();

      setPrediction(preds.dataSync()[0].toString());
    }
  };

  return (
    <div>
      <h2>Digit Recognition</h2>
      <button type="button" onClick={run}>
        Train model
      </button>
      <div className={styles.canvasContainer}>
        {/* @ts-ignore */}
        <DigitCanvas scale={CANVAS_SCALE} ref={canvasRef} />
        {prediction && (
          <span className={styles.prediction}>
            Prediction: <strong>{prediction}</strong>
          </span>
        )}
      </div>
      {model && (
        <button type="button" onClick={recognize}>
          Recognize
        </button>
      )}
    </div>
  );
}

export default DigitRecognizer;

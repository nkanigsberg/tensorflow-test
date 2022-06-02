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

/** This component trains a model to recognize handwritten digits with a convolutional neural network, and then uses that model to predict the digit drawn by the user.
 *
 * Note: This is largely based on this TensorFlow.js tutorial: https://www.tensorflow.org/js/tutorials/training/handwritten_digit_cnn
 */
function DigitRecognizer() {
  const [prediction, setPrediction] = useState<string>("");
  const [model, setModel] = useState<tf.Sequential>();
  const [trainDataSize, setTrainDataSize] = useState<number>(5500);
  const [testDataSize, setTestDataSize] = useState<number>(1000);
  const [batchSize, setBatchSize] = useState<number>(512);
  const [epochs, setEpochs] = useState<number>(10);
  const [learningRate, setLearningRate] = useState<number>(0.001);

  const canvasRef = useRef<HTMLCanvasElement>(null);

  const run = async () => {
    const data = new MnistData();
    await data.load();
    await showExamples(data);
    const model = getModel(learningRate);
    setModel(model);

    tfvis.show.modelSummary(
      { name: "Model Architecture", tab: "Model" },
      model
    );

    await train(model, data, {
      batchSize,
      epochs,
      trainDataSize,
      testDataSize,
    });

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
    <div className={styles.digitRecognizer}>
      <h2>Digit Recognition</h2>
      <div className={styles.controls}>
        <p>Size of dataset: 60000</p>
        <label>
          Train data size:
          <input
            type="number"
            onChange={(e) => setTrainDataSize(parseInt(e.target.value))}
            value={trainDataSize}
          />
        </label>
        <label>
          Test data size:
          <input
            type="number"
            onChange={(e) => setTestDataSize(parseInt(e.target.value))}
            value={testDataSize}
          />
        </label>
        <label>
          Batch size:
          <input
            type="number"
            onChange={(e) => setBatchSize(parseInt(e.target.value))}
            value={batchSize}
          />
        </label>
        <label>
          Epochs:
          <input
            type="number"
            onChange={(e) => setEpochs(parseInt(e.target.value))}
            value={epochs}
          />
        </label>
        <label>
          Learning rate:
          <input
            type="number"
            onChange={(e) => setLearningRate(parseFloat(e.target.value))}
            value={learningRate}
          />
        </label>
      </div>

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
        <button
          type="button"
          onClick={recognize}
          className={styles.recognizeButton}
        >
          Recognize
        </button>
      )}
    </div>
  );
}

export default DigitRecognizer;

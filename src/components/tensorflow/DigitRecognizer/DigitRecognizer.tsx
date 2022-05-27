import { useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";

import { MnistData } from "./data";

async function showExamples(data: MnistData) {
  // Create a container in the visor
  const surface = tfvis
    .visor()
    .surface({ name: "Input Data Examples", tab: "Input Data" });

  // Get the examples
  const examples = data.nextTestBatch(20);
  const numExamples = examples.xs.shape[0];

  // Create a canvas element to render each example
  for (let i = 0; i < numExamples; i++) {
    const imageTensor: tf.Tensor<tf.Rank.R2> = tf.tidy(() => {
      // Reshape the image to 28x28 px
      return examples.xs
        .slice([i, 0], [1, examples.xs.shape[1]])
        .reshape([28, 28, 1]);
    });

    const canvas = document.createElement("canvas");
    canvas.width = 28;
    canvas.height = 28;
    canvas.setAttribute("style", "margin: 4px;");
    await tf.browser.toPixels(imageTensor, canvas);
    surface.drawArea.appendChild(canvas);

    imageTensor.dispose();
  }
}

function DigitRecognizer() {
  useEffect(() => {
    (async () => {
      const data = new MnistData();
      await data.load();
      await showExamples(data);
    })();
  }, []);

  return (
    <div>
      <h2>Digit Recognition</h2>
    </div>
  );
}

export default DigitRecognizer;

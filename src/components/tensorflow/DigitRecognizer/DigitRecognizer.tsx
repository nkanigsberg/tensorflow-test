import * as tfvis from "@tensorflow/tfjs-vis";

import {
  showExamples,
  getModel,
  train,
  showAccuracy,
  showConfusion,
} from "./helpers";

import { MnistData } from "./data";

function DigitRecognizer() {
  const run = async () => {
    const data = new MnistData();
    await data.load();
    await showExamples(data);

    const model = getModel();
    tfvis.show.modelSummary(
      { name: "Model Architecture", tab: "Model" },
      model
    );

    await train(model, data);

    await showAccuracy(model, data);
    await showConfusion(model, data);
  };

  return (
    <div>
      <h2>Digit Recognition</h2>
      <button type="button" onClick={run}>
        Run
      </button>
    </div>
  );
}

export default DigitRecognizer;

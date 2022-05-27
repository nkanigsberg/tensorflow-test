import { DataPredictor } from "./components/tensorflow/DataPredictor";
import { DigitRecognizer } from "./components/tensorflow/DigitRecognizer";

import styles from "./App.module.css";

function App() {
  return (
    <div className={styles.wrapper}>
      <h1>TensorFlow demo</h1>
      <DataPredictor />
      <DigitRecognizer />
    </div>
  );
}

export default App;

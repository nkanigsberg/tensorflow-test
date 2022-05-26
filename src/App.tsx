import { DataPredictor } from "./components/tensorflow/DataPredictor";

import styles from "./App.module.css";

function App() {
  return (
    <div className={styles.wrapper}>
      <h1>TensorFlow demo</h1>
      <DataPredictor />
    </div>
  );
}

export default App;

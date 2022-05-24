import { Classifier } from "./components/tensorflow/Classifier";

import styles from "./App.module.css";

function App() {
  return (
    <div className={styles.wrapper}>
      <h1>Image classifier demo</h1>
      <Classifier />
    </div>
  );
}

export default App;

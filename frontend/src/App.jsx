import React from "react";
import PlantDiseasePredictor from "./components/PlantDiseasePredictor";

function App() {
  return (
    <div className="App d-flex align-items-center justify-content-center min-vh-100 bg-light">
        <h1 className="text-center my-4">Plant Disease Detector</h1>
        <PlantDiseasePredictor />
    </div>
  );
}

export default App;

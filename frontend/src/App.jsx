// src/App.jsx

import React from "react";
import PlantDiseasePredictor from "./components/PlantDiseasePredictor";
import "./App.css"; // This should now be an empty file

function App() {
  // DO NOT add any <h1> or <div> tags here.
  // The PlantDiseasePredictor component handles the entire page.
  return (
    <PlantDiseasePredictor />
  );
}

export default App;
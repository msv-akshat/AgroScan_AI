// App.jsx

import React from "react";
import PlantDiseasePredictor from "./components/PlantDiseasePredictor";
import "./App.css"; // Assuming you have a CSS file for custom styles

function App() {
  return (
    // min-vh-100 ensures the container takes the full viewport height.
    // bg-light is used for a soft background.
    <div className="App d-flex flex-column align-items-center justify-content-center min-vh-100 bg-light">
      
      {/* Centered App Title with "AgroScan" branding */}
      <h1 className="display-4 text-center fw-bold mb-4" style={{ color: '#0f7d0f' }}>
        <i className="bi bi-tree-fill me-2"></i> AgroScan
      </h1>
      
      {/* Component is wrapped in the centered parent div */}
      <PlantDiseasePredictor />
      
    </div>
  );
}

export default App;

// You will also need to add the Bootstrap Icons library for the tree icon:
// npm install bootstrap-icons
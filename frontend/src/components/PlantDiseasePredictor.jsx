// src/components/PlantDiseasePredictor.jsx

import React, { useState, useRef } from "react";
import PlantSelector from "./PlantSelector";
import ImageUploader from "./ImageUploader";
import PredictionResult from "./PredictionResult";
import TopKModal from "./TopKModal";
import Loader from "./Loader";

const PlantDiseasePredictor = () => {
  const [file, setFile] = useState(null);
  const [plant, setPlant] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [topk, setTopk] = useState([]);
  const [showTopk, setShowTopk] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const fileInputRef = useRef(null);

  const BACKEND_URL =
    import.meta.env.VITE_BACKEND_URL ||
    "https://fecil5ew47tajxpqobh45lar640pwlmv.lambda-url.us-east-1.on.aws/";

  const handleFileChange = (e) => {
    const f = e.target.files?.[0] || null;
    setFile(f);
    setPreviewUrl(f ? URL.createObjectURL(f) : null);
  };

  const fileToBase64 = (file) =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result.split(",")[1]);
      reader.onerror = reject;
    });

  const ensureFile = () => {
    if (!(file instanceof File)) {
      fileInputRef.current && (fileInputRef.current.value = "");
      alert("Please select an image again.");
      return false;
    }
    return true;
  };

  const callApi = async (mode) => {
    if (!ensureFile() || !plant) return alert("Please select a plant.");
    setIsLoading(true);
    try {
      const base64Image = await fileToBase64(file);
      const body = JSON.stringify({ plant, image: base64Image, mode });
      const res = await fetch(BACKEND_URL, { method: "POST", body });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error || `HTTP ${res.status}`);

      if (mode === "predict") {
        setPrediction({
          prediction: data.prediction,
          confidence: data.confidence,
        });
        setTopk([]);
        setShowTopk(false);
      } else if (mode === "topk") {
        setTopk(data.topk || []);
        setPrediction(null);
        setShowTopk(true);
      }
    } catch (err) {
      console.error(err);
      alert(`${mode} failed`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="d-flex flex-column align-items-center min-vh-100 py-4 bg-light">
      <div className="card shadow-lg rounded-4 p-4" style={{ maxWidth: "600px", width: "100%" }}>
        <h1 className="text-center fw-bold mb-4 text-success">
          <i className="bi bi-tree-fill me-2"></i> AgroScan
        </h1>

        <div className="text-center mb-4 p-3 rounded-4 bg-success text-white">
          <h2 className="mb-0 h4">🌱 Plant Disease Detector</h2>
          <small className="d-block">AI-powered analysis</small>
        </div>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            callApi("predict");
          }}
        >
          <PlantSelector plant={plant} onChange={(e) => setPlant(e.target.value)} />
          <ImageUploader
            file={file}
            previewUrl={previewUrl}
            onFileChange={handleFileChange}
            inputRef={fileInputRef}
          />

          <div className="d-flex justify-content-center gap-3 mb-3 flex-wrap">
            <button
              type="submit"
              className="btn btn-success px-4 fw-semibold"
              disabled={isLoading}
            >
              🔍 Predict
            </button>
            <button
              type="button"
              className="btn btn-outline-success px-4 fw-semibold"
              onClick={() => callApi("topk")}
              disabled={isLoading}
            >
              📊 Top-5
            </button>
          </div>
        </form>

        {isLoading && <Loader />}
        {prediction && <PredictionResult plant={plant} prediction={prediction} />}
      </div>

      {showTopk && <TopKModal topk={topk} onClose={() => setShowTopk(false)} />}
    </div>
  );
};

export default PlantDiseasePredictor;

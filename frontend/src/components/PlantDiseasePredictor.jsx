import React, { useState, useRef } from "react";
import { createRoot } from 'react-dom/client'
import '../index.css'
import App from '../App.jsx'

const PlantDiseasePredictor = () => {
  const [file, setFile] = useState(null);
  const [plant, setPlant] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [topk, setTopk] = useState([]);
  const [showTopk, setShowTopk] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const fileInputRef = useRef(null);

  const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5000";

  const handleFileChange = (e) => {
    const f = e.target.files && e.target.files.length > 0 ? e.target.files[0] : null;
    setFile(f);

    if (f) {
      setPreviewUrl(URL.createObjectURL(f));
    } else {
      setPreviewUrl(null);
    }
  };

  const handlePlantChange = (e) => setPlant(e.target.value);

  const ensureFile = () => {
    if (!(file && file instanceof File)) {
      if (fileInputRef.current) fileInputRef.current.value = "";
      alert("Please select an image again.");
      return false;
    }
    return true;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!ensureFile()) return;
    if (!plant) {
      alert("Please select a plant.");
      return;
    }
    if (!(file instanceof File)) {
      alert("Please select an image again.");
      return;
    }

    setIsLoading(true);
    const formData = new FormData();
    formData.append("plant", plant);
    formData.append("image", file);

    try {
      const res = await fetch(`${BACKEND_URL}/predict`, { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error || `HTTP ${res.status}`);
      setPrediction({
        prediction: data.prediction ?? data.predicted_class ?? data.predictedClass,
        confidence: data.confidence,
      });
      setTopk([]);
      setShowTopk(false);
    } catch (err) {
      console.error(err);
      alert("Prediction failed");
      setPrediction(null);
    } finally {
      setIsLoading(false);
    }
  };

  const handleTopk = async () => {
    if (!ensureFile()) return;
    if (!plant) {
      alert("Please select a plant.");
      return;
    }

    setIsLoading(true);
    const formData = new FormData();
    formData.append("plant", plant);
    formData.append("image", file);

    try {
      const res = await fetch(`${BACKEND_URL}/topk`, { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error || `HTTP ${res.status}`);
      setTopk(data.topk || []);
      setPrediction(null);
      setShowTopk(true);
    } catch (err) {
      console.error(err);
      alert("Top‑5 failed");
      setShowTopk(false);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container mt-5">
      <div className="row justify-content-center">
        <div className="col-md-8">
          <div className="card p-4 shadow">
            <h2 className="card-title text-center mb-4">Plant Disease Predictor 🌱</h2>
            <form onSubmit={handleSubmit}>
              <div className="mb-3">
                <label htmlFor="plantSelect" className="form-label">Select Plant</label>
                <select id="plantSelect" className="form-select" value={plant} onChange={handlePlantChange}>
                  <option value="">-- Select Plant --</option>
                  <option value="tomato">Tomato</option>
                  <option value="potato">Potato</option>
                  <option value="maize">Maize</option>
                  <option value="grape">Grape</option>
                  <option value="apple">Apple</option>
                </select>
              </div>

              <div className="mb-3">
                <label htmlFor="imageUpload" className="form-label">Upload Image</label>
                <input
                  id="imageUpload"
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  className="form-control"
                  onClick={(ev) => (ev.currentTarget.value = "")}
                  onChange={handleFileChange}
                />
              </div>

              {previewUrl && (
                <div className="text-center mb-3">
                  <p className="fw-bold">Preview:</p>
                  <img src={previewUrl} alt="Selected preview" className="img-fluid rounded border" style={{ maxHeight: "300px" }} />
                </div>
              )}

              {file && <div className="text-center mb-3 text-muted">Selected: {file.name}</div>}

              <div className="d-grid gap-2 d-md-flex justify-content-md-center mb-3">
                <button type="submit" className="btn btn-primary" disabled={isLoading}>Predict</button>
                <button type="button" onClick={handleTopk} className="btn btn-secondary" disabled={isLoading}>Show Top‑5</button>
              </div>
            </form>

            {isLoading && (
              <div className="d-flex justify-content-center mt-4">
                <div className="spinner-border text-primary" role="status">
                  <span className="visually-hidden">Loading...</span>
                </div>
              </div>
            )}

            {prediction && (
              <div className="card mt-4 p-3 bg-light">
                <h4 className="card-title text-center">Prediction Result</h4>
                <p className="mb-1"><strong>Plant:</strong> {plant}</p>
                <p className="mb-1"><strong>Disease:</strong> {prediction.prediction}</p>
                {prediction.confidence != null && (
                  <p className="mb-0"><strong>Confidence:</strong> {(prediction.confidence * 100).toFixed(2)}%</p>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {showTopk && (
        <>
          <div className="modal fade show d-block" tabIndex="-1" role="dialog" aria-labelledby="topkModalLabel" aria-hidden="true">
            <div className="modal-dialog modal-dialog-centered">
              <div className="modal-content">
                <div className="modal-header">
                  <h5 className="modal-title" id="topkModalLabel">Top‑5 Predictions</h5>
                  <button type="button" className="btn-close" onClick={() => setShowTopk(false)} aria-label="Close"></button>
                </div>
                <div className="modal-body">
                  <ul className="list-group list-group-flush">
                    {topk.map((t, i) => (
                      <li key={i} className="list-group-item d-flex justify-content-between align-items-center">
                        {t.class}
                        <span className="badge bg-primary rounded-pill">{(t.confidence * 100).toFixed(2)}%</span>
                      </li>
                    ))}
                  </ul>
                </div>
                <div className="modal-footer">
                  <button type="button" className="btn btn-secondary" onClick={() => setShowTopk(false)}>Close</button>
                </div>
              </div>
            </div>
          </div>
          <div className="modal-backdrop fade show"></div>
        </>
      )}
    </div>
  );
};

export default PlantDiseasePredictor;

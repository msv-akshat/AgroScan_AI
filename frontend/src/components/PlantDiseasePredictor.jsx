import React, { useState, useRef } from "react";
import "../index.css";

const PlantDiseasePredictor = () => {
  const [file, setFile] = useState(null);
  const [plant, setPlant] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [topk, setTopk] = useState([]);
  const [showTopk, setShowTopk] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const fileInputRef = useRef(null);
  const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://13.48.56.255:5000";

  const handleFileChange = (e) => {
    const f = e.target.files?.[0] || null;
    setFile(f);
    setPreviewUrl(f ? URL.createObjectURL(f) : null);
  };

  const handlePlantChange = (e) => setPlant(e.target.value);

  const ensureFile = () => {
    if (!(file instanceof File)) {
      fileInputRef.current && (fileInputRef.current.value = "");
      alert("Please select an image again.");
      return false;
    }
    return true;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!ensureFile() || !plant) return alert("Please select a plant.");

    setIsLoading(true);
    const formData = new FormData();
    formData.append("plant", plant);
    formData.append("image", file);

    try {
      const res = await fetch(`${BACKEND_URL}/predict`, { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error || `HTTP ${res.status}`);
      setPrediction({ prediction: data.prediction ?? data.predicted_class ?? data.predictedClass, confidence: data.confidence });
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
    if (!ensureFile() || !plant) return alert("Please select a plant.");

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
    <div className="container my-5">
      <div className="row justify-content-center">
        <div className="col-lg-7 col-md-8 col-sm-10">
          <div className="card shadow-lg p-4 border-0 rounded-4">
            <form onSubmit={handleSubmit}>
              <div className="mb-3">
                <label htmlFor="plantSelect" className="form-label fw-bold">Select Plant</label>
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
                <label htmlFor="imageUpload" className="form-label fw-bold">Upload Image</label>
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
                  <p className="fw-semibold">Preview</p>
                  <img src={previewUrl} alt="preview" className="img-fluid rounded-4 border shadow-sm" style={{ maxHeight: "300px" }} />
                </div>
              )}

              {file && <p className="text-center text-muted mb-3">Selected: {file.name}</p>}

              <div className="d-flex justify-content-center gap-3 mb-3 flex-wrap">
                <button type="submit" className="btn btn-success px-4" disabled={isLoading}>Predict</button>
                <button type="button" className="btn btn-outline-success px-4" onClick={handleTopk} disabled={isLoading}>Top‑5</button>
              </div>
            </form>

            {isLoading && (
              <div className="text-center my-4">
                <div className="spinner-border text-success" role="status"></div>
              </div>
            )}

            {prediction && (
              <div className="card mt-4 p-3 bg-light rounded-4 shadow-sm border">
                <h4 className="text-center text-success mb-3">Prediction Result</h4>
                <p><strong>Plant:</strong> {plant}</p>
                <p><strong>Disease:</strong> {prediction.prediction}</p>
                {prediction.confidence != null && <p><strong>Confidence:</strong> {(prediction.confidence * 100).toFixed(2)}%</p>}
              </div>
            )}
          </div>
        </div>
      </div>

      {showTopk && (
        <>
          <div className="modal fade show d-block" tabIndex="-1">
            <div className="modal-dialog modal-dialog-centered">
              <div className="modal-content rounded-4 shadow-lg border-0">
                <div className="modal-header bg-success text-white rounded-top-4">
                  <h5 className="modal-title">Top‑5 Predictions</h5>
                  <button type="button" className="btn-close btn-close-white" onClick={() => setShowTopk(false)}></button>
                </div>
                <div className="modal-body">
                  <ul className="list-group list-group-flush">
                    {topk.map((t, i) => (
                      <li key={i} className="list-group-item d-flex justify-content-between align-items-center">
                        {t.class}
                        <span className="badge bg-success rounded-pill">{(t.confidence * 100).toFixed(2)}%</span>
                      </li>
                    ))}
                  </ul>
                </div>
                <div className="modal-footer">
                  <button type="button" className="btn btn-outline-success" onClick={() => setShowTopk(false)}>Close</button>
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

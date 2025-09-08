const express = require("express");
const multer = require("multer");
const cors = require("cors");
const fs = require("fs");
const axios = require("axios");
const FormData = require("form-data");
const path = require("path");

const app = express();
app.use(cors());

// temp upload dir for Multer
const upload = multer({ dest: "uploads/" });

// Get the Flask API URL from an environment variable, with a fallback for local testing
const FLASK_API_URL = process.env.FLASK_API_URL || "http://localhost:8000";

// Predict: expects multipart with fields: plant (text) and image (file)
app.post("/predict", upload.single("image"), async (req, res) => {
  try {
    const plant = req.body?.plant || "";

    if (!req.file) return res.status(400).json({ error: "Image is required (field name must be 'image')." });
    if (!plant) return res.status(400).json({ error: "Plant is required (field name must be 'plant')." });

    // forward to Flask as multipart; Flask expects 'file' and optional 'plant'
    const form = new FormData();
    form.append("plant", plant);
    form.append("file", fs.createReadStream(req.file.path));

    const response = await axios.post(`${FLASK_API_URL}/predict`, form, {
      headers: form.getHeaders(),
      maxContentLength: Infinity,
      maxBodyLength: Infinity,
    });

    const out = response.data;
    fs.unlink(req.file.path, () => {});

    return res.json({
      prediction: out.predicted_class,
      confidence: out.confidence,
      filtered: out.filtered_predictions || null
    });
  } catch (err) {
    console.error(err?.response?.data || err.message);
    return res.status(500).json({ error: "Prediction failed" });
  }
});

app.post("/topk", upload.single("image"), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: "Image is required" });
  const plant = req.body?.plant || "";
  if (!plant) return res.status(400).json({ error: "Plant is required (field name must be 'plant')." });
  try {
    const form = new FormData();
    form.append("file", fs.createReadStream(req.file.path));
    form.append("plant", plant);
    const r = await axios.post(`${FLASK_API_URL}/topk`, form, {
      headers: form.getHeaders(),
      maxContentLength: Infinity,
      maxBodyLength: Infinity,
    });
    fs.unlink(req.file.path, () => {});
    return res.json({ topk: r.data.topk || [] });
  } catch (e) {
    console.error(e?.response?.data || e.message);
    return res.status(500).json({ error: "Top‑k failed" });
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Node backend listening on port ${PORT}`));

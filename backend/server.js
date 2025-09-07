const express = require("express");
const multer = require("multer");
const cors = require("cors");
const fs = require("fs");
const axios = require("axios");
const FormData = require("form-data");

const app = express();
app.use(cors());

// temp upload dir for Multer
const upload = multer({ dest: "uploads/" });

// Predict: expects multipart with fields: plant (text) and image (file)
app.post("/predict", upload.single("image"), async (req, res) => {
  try {
    const plant = req.body?.plant || "";

    console.log("Multer body:", req.body); // should NOT contain 'image' when file parsed [3]
    console.log("Multer file:", req.file && { fieldname: req.file.fieldname, originalname: req.file.originalname, path: req.file.path }); // debug [3]

    if (!req.file) return res.status(400).json({ error: "Image is required (field name must be 'image')." }); // guard [3]
    if (!plant) return res.status(400).json({ error: "Plant is required (field name must be 'plant')." }); // guard [3]

    // forward to Flask as multipart; Flask expects 'file' and optional 'plant'
    const form = new FormData();
    form.append("plant", plant);
    form.append("file", fs.createReadStream(req.file.path)); // key 'file' for Flask [10]

    const response = await axios.post("http://127.0.0.1:8000/predict", form, {
      headers: form.getHeaders(),
      maxContentLength: Infinity,
      maxBodyLength: Infinity,
    });

    const out = response.data; // { predicted_class, confidence, filtered_predictions? } [10]
    fs.unlink(req.file.path, () => {}); // cleanup temp file regardless [3]

    return res.json({
      prediction: out.predicted_class,
      confidence: out.confidence,
      filtered: out.filtered_predictions || null
    });
  } catch (err) {
    console.error(err?.response?.data || err.message); // debug [3]
    return res.status(500).json({ error: "Prediction failed" }); // error response [3]
  }
});

app.post("/topk", upload.single("image"), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: "Image is required" }); // guard [3]
  const plant = req.body?.plant || "";
  if (!plant) return res.status(400).json({ error: "Plant is required (field name must be 'plant')." }); // guard [3]
  try {
    const form = new FormData();
    form.append("file", fs.createReadStream(req.file.path)); // key 'file' [10]
    form.append("plant", plant); // Add this line to forward the plant parameter.
    const r = await axios.post("http://127.0.0.1:8000/topk", form, {
      headers: form.getHeaders(),
      maxContentLength: Infinity,
      maxBodyLength: Infinity,
    });
    fs.unlink(req.file.path, () => {}); // cleanup [3]
    return res.json({ topk: r.data.topk || [] }); // normalize [3]
  } catch (e) {
    console.error(e?.response?.data || e.message); // debug [3]
    return res.status(500).json({ error: "Top‑k failed" }); // error [3]
  }
});

app.listen(5000, () => console.log("Node backend on 5000")); // start server [3]

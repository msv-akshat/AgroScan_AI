const PredictionResult = ({ plant, prediction }) => (
  <div className="card mt-4 p-3 bg-light rounded-4 shadow-sm border">
    <h4 className="text-center text-success mb-3">Prediction Result</h4>
    <p><strong>Plant:</strong> {plant}</p>
    <p><strong>Disease:</strong> {prediction.prediction}</p>
    {prediction.confidence != null && (
      <p><strong>Confidence:</strong> {(prediction.confidence * 100).toFixed(2)}%</p>
    )}
  </div>
);

export default PredictionResult;

import { CheckCircleFill, ExclamationTriangleFill } from "react-bootstrap-icons";

const PredictionResult = ({ plant, prediction }) => {
  const isHealthy = prediction.prediction.toLowerCase().includes("healthy");

  return (
    <div
      className="card mt-4 shadow-lg border-0 rounded-4"
      style={{
        background: isHealthy
          ? "linear-gradient(135deg, #f9fff9, #e9fce9)"  // light green for healthy
          : "linear-gradient(135deg, #fff5f5, #fdeaea)"  // light red for diseased
      }}
    >
      <div
        className="card-header d-flex align-items-center rounded-top-4"
        style={{
          background: isHealthy
            ? "linear-gradient(135deg, #28a745, #1e7e34)" // green header
            : "linear-gradient(135deg, #dc3545, #a71d2a)", // red header
          color: "white",
          fontWeight: "600",
        }}
      >
        {isHealthy ? (
          <CheckCircleFill size={22} className="me-2" />
        ) : (
          <ExclamationTriangleFill size={22} className="me-2" />
        )}
        Prediction Result
      </div>

      <div className="card-body">
        <p className="mb-2">
          <strong>🌿 Plant:</strong> {plant}
        </p>
        <p className="mb-2">
          <strong>🦠 Disease:</strong>{" "}
          <span
            className={`fw-semibold ${
              isHealthy ? "text-success" : "text-danger"
            }`}
          >
            {prediction.prediction}
          </span>
        </p>
        {prediction.confidence != null && (
          <p className="mb-0">
            <strong>📊 Confidence:</strong>{" "}
            <span className={`badge fs-6 ${isHealthy ? "bg-success" : "bg-danger"}`}>
              {(prediction.confidence * 100).toFixed(2)}%
            </span>
          </p>
        )}
      </div>
    </div>
  );
};

export default PredictionResult;

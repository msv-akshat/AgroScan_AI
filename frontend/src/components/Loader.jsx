import { ArrowRepeat } from "react-bootstrap-icons";

const Loader = () => (
  <div className="text-center my-5">
    {/* Animated Glow Spinner */}
    <div
      className="d-inline-flex align-items-center justify-content-center rounded-circle shadow"
      style={{
        width: "80px",
        height: "80px",
        background:
          "radial-gradient(circle at center, rgba(40,167,69,0.15), transparent 70%)",
        animation: "pulse 1.5s infinite",
      }}
    >
      <ArrowRepeat
        className="text-success"
        size={40}
        style={{ animation: "spin 1.2s linear infinite" }}
      />
    </div>
    <p className="mt-3 fw-semibold text-success">Analyzing image...</p>

    {/* CSS Animations */}
    <style>
      {`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        @keyframes pulse {
          0% { box-shadow: 0 0 0 0 rgba(40,167,69, 0.4); }
          70% { box-shadow: 0 0 0 15px rgba(40,167,69, 0); }
          100% { box-shadow: 0 0 0 0 rgba(40,167,69, 0); }
        }
      `}
    </style>
  </div>
);

export default Loader;

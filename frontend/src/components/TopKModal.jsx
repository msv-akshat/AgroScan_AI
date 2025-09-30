import { TrophyFill } from "react-bootstrap-icons";

const TopKModal = ({ topk, onClose }) => (
  <>
    <div className="modal fade show d-block" tabIndex="-1">
      <div className="modal-dialog modal-dialog-centered">
        <div className="modal-content rounded-4 shadow-lg border-0">
          {/* Modal Header */}
          <div
            className="modal-header text-white rounded-top-4"
            style={{
              background: "linear-gradient(135deg, #0f7d0f, #1ca21c)",
            }}
          >
            <h5 className="modal-title d-flex align-items-center">
              <TrophyFill size={20} className="me-2" /> Top-5 Predictions
            </h5>
            <button
              type="button"
              className="btn-close btn-close-white"
              onClick={onClose}
            ></button>
          </div>

          {/* Modal Body */}
          <div className="modal-body">
            {topk.length === 0 ? (
              <p className="text-center text-muted">No predictions available.</p>
            ) : (
              <ul className="list-group list-group-flush">
                {topk.map((t, i) => (
                  <li
                    key={i}
                    className="list-group-item border-0 py-3"
                    style={{ backgroundColor: "transparent" }}
                  >
                    <div className="d-flex justify-content-between">
                      <span className="fw-semibold">{t.class}</span>
                      <span className="text-success fw-bold">
                        {(t.confidence * 100).toFixed(2)}%
                      </span>
                    </div>
                    {/* Confidence Progress Bar */}
                    <div className="progress mt-2" style={{ height: "8px" }}>
                      <div
                        className="progress-bar bg-success"
                        role="progressbar"
                        style={{ width: `${t.confidence * 100}%` }}
                      ></div>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>

          {/* Modal Footer */}
          <div className="modal-footer">
            <button
              type="button"
              className="btn btn-outline-success rounded-pill px-4"
              onClick={onClose}
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
    <div className="modal-backdrop fade show"></div>
  </>
);

export default TopKModal;

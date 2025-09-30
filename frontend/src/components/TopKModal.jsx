const TopKModal = ({ topk, onClose }) => (
  <>
    <div className="modal fade show d-block" tabIndex="-1">
      <div className="modal-dialog modal-dialog-centered">
        <div className="modal-content rounded-4 shadow-lg border-0">
          <div className="modal-header bg-success text-white rounded-top-4">
            <h5 className="modal-title">Top-5 Predictions</h5>
            <button
              type="button"
              className="btn-close btn-close-white"
              onClick={onClose}
            ></button>
          </div>
          <div className="modal-body">
            <ul className="list-group list-group-flush">
              {topk.map((t, i) => (
                <li
                  key={i}
                  className="list-group-item d-flex justify-content-between align-items-center"
                >
                  {t.class}
                  <span className="badge bg-success rounded-pill">
                    {(t.confidence * 100).toFixed(2)}%
                  </span>
                </li>
              ))}
            </ul>
          </div>
          <div className="modal-footer">
            <button
              type="button"
              className="btn btn-outline-success"
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

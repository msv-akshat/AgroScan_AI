import { useState } from "react";
import { XCircleFill } from "react-bootstrap-icons"; // Bootstrap Icons

const ImageUploader = ({ file, previewUrl, onFileChange, inputRef }) => {
  const [dragOver, setDragOver] = useState(false);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const event = { target: { files: [e.dataTransfer.files[0]] } };
      onFileChange(event);
    }
  };

  return (
    <div className="mb-4">
      <label htmlFor="imageUpload" className="form-label fw-bold">
        Upload Image
      </label>

      {/* Drag-and-drop zone */}
      <div
        className={`border border-2 rounded-4 p-4 text-center shadow-sm ${
          dragOver ? "border-success bg-light" : "border-secondary"
        }`}
        style={{ cursor: "pointer", transition: "0.3s" }}
        onClick={() => inputRef.current.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
      >
        {!file ? (
          <div className="text-muted">
            <p className="mb-1">
              <strong>Drag & Drop</strong> an image here
            </p>
            <small>or click to browse</small>
          </div>
        ) : (
          previewUrl && (
            <div className="position-relative d-inline-block">
              <img
                src={previewUrl}
                alt="preview"
                className="img-fluid rounded-4 border shadow-sm"
                style={{ maxHeight: "250px" }}
              />
              {/* Easy Close Button */}
              <button
                type="button"
                className="btn p-0 border-0 position-absolute top-0 end-0 m-1"
                onClick={() => {
                  inputRef.current.value = "";
                  onFileChange({ target: { files: [] } });
                }}
              >
                <XCircleFill size={28} className="text-danger bg-white rounded-circle" />
              </button>
              <p className="mt-2 text-muted small">{file.name}</p>
            </div>
          )
        )}
      </div>

      {/* Hidden input */}
      <input
        id="imageUpload"
        ref={inputRef}
        type="file"
        accept="image/*"
        className="d-none"
        onClick={(ev) => (ev.currentTarget.value = "")}
        onChange={onFileChange}
      />
    </div>
  );
};

export default ImageUploader;

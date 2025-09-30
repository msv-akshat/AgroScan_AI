const ImageUploader = ({ file, previewUrl, onFileChange, inputRef }) => (
  <div className="mb-3">
    <label htmlFor="imageUpload" className="form-label fw-bold">Upload Image</label>
    <input
      id="imageUpload"
      ref={inputRef}
      type="file"
      accept="image/*"
      className="form-control"
      onClick={(ev) => (ev.currentTarget.value = "")}
      onChange={onFileChange}
    />

    {previewUrl && (
      <div className="text-center mt-3">
        <p className="fw-semibold">Preview</p>
        <img
          src={previewUrl}
          alt="preview"
          className="img-fluid rounded-4 border shadow-sm"
          style={{ maxHeight: "300px" }}
        />
      </div>
    )}

    {file && (
      <p className="text-center text-muted mt-2">
        Selected: {file.name}
      </p>
    )}
  </div>
);

export default ImageUploader;

import PLANTS from "../config/plants";

const PlantSelector = ({ plant, onChange }) => (
  <div className="mb-3">
    <label htmlFor="plantSelect" className="form-label fw-bold">Select Plant</label>
    <select
      id="plantSelect"
      className="form-select"
      value={plant}
      onChange={onChange}
    >
      <option value="">-- Select Plant --</option>
      {PLANTS.map((p) => (
        <option key={p.value} value={p.value}>{p.label}</option>
      ))}
    </select>
  </div>
);

export default PlantSelector;

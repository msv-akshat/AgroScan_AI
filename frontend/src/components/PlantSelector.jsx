import { Flower1 } from "react-bootstrap-icons";
import PLANTS from "../config/plants";

const PlantSelector = ({ plant, onChange }) => (
  <div className="mb-4">
    <label
      htmlFor="plantSelect"
      className="form-label fw-bold d-flex align-items-center"
    >
      <Flower1 size={20} className="me-2 text-success" />
      Select Plant
    </label>

    <div className="input-group">
      <select
        id="plantSelect"
        className="form-select rounded-pill shadow-sm border-0 px-3 py-2"
        value={plant}
        onChange={onChange}
        style={{
          background: "linear-gradient(135deg, #f9fff9, #e9fce9)",
          fontWeight: "500",
        }}
      >
        <option value="">-- Select Plant --</option>
        {PLANTS.map((p) => (
          <option key={p.value} value={p.value}>
            {p.label}
          </option>
        ))}
      </select>
    </div>
  </div>
);

export default PlantSelector;

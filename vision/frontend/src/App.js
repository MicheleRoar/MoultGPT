// App.js — MoultVision (YOLO + XGBoost + optional FastSAM)
import React, { useState, useRef } from "react";
import axios from "axios";
import "./App.css";

// Taxon groups (one-hot encoded on backend)
const taxonOptions = [
  { label: "Crustacea",   value: 0 },
  { label: "Hexapoda",    value: 1 },
  { label: "Chelicerata", value: 2 },
  { label: "Myriapoda",   value: 3 },
];

// Backend URL (can be overridden via .env)
const API_URL = process.env.REACT_APP_API_URL || "http://127.0.0.1:5001";

// Fixed canvas size for display (image + overlays)
const CANVAS = 640;

// Local fallback for stage labels (index-based)
const IDX2LABEL = ["post-moult", "moulting", "exuviae"];

export default function App() {
  const [image, setImage] = useState(null);              // raw File
  const [taxonId, setTaxonId] = useState("");            // selected taxon (0–3)
  const [resData, setResData] = useState(null);          // backend response
  const [imageLoading, setImageLoading] = useState(false);
  const [imagePreview, setImagePreview] = useState(null);// local preview URL
  const [error, setError] = useState("");

  // Natural image size (used to rescale boxes onto fixed canvas)
  const [naturalSize, setNaturalSize] = useState({ w: 224, h: 224 });
  const imgRef = useRef(null);

  // UI toggles
  const [showMask, setShowMask] = useState(true);        // show/hide segmentation masks

  // Handle image selection
  const onImageSelect = (e) => {
    const f = e.target.files?.[0];
    setError("");
    setResData(null);
    setImagePreview(null);

    if (!f) {
      setImage(null);
      return;
    }

    setImage(f);
    setImagePreview(URL.createObjectURL(f));
  };

  // Call backend /predict_image
  const onPredict = async () => {
    if (!image) {
      setError("Select an image first.");
      return;
    }
    if (taxonId === "") {
      setError("Select a taxon.");
      return;
    }

    setError("");
    setImageLoading(true);
    setResData(null);

    try {
      const fd = new FormData();
      fd.append("image", image);
      fd.append("taxon_id", taxonId);
      fd.append("use_seg", showMask ? "1" : "0");

      const res = await axios.post(`${API_URL}/predict_image`, fd);
      setResData(res.data);
    } catch (e) {
      const serverMsg = e?.response?.data?.error || e?.message || "Prediction failed or backend not reachable.";
      setError(`⚠️ ${serverMsg}`);
    } finally {
      setImageLoading(false);
    }
  };

  // Reset UI state
  const onClear = () => {
    setImage(null);
    setTaxonId("");
    setResData(null);
    setImagePreview(null);
    setError("");
    setShowMask(true);

    const i = document.getElementById("image-input");
    if (i) i.value = null;
  };

  // Map stage index/string to human-readable label
  function stageToLabel(stage, classLabels) {
    if (stage == null) return null;

    // If it's already a non-numeric string, just return it
    if (typeof stage === "string" && !/^\d+$/.test(stage)) return stage;

    const idx = Number(stage);
    if (!Number.isFinite(idx)) return String(stage);

    // 1) Prefer backend-provided class_labels
    if (Array.isArray(classLabels) && classLabels[idx]) return classLabels[idx];

    // 2) Fallback to local mapping
    if (IDX2LABEL[idx]) return IDX2LABEL[idx];

    // 3) Final fallback: raw value
    return String(stage);
  }

  const classLabels = resData?.class_labels || null;
  const stageLabel = stageToLabel(resData?.stage, classLabels);
  const stageConfPct =
    resData?.stage_confidence != null
      ? `${(resData.stage_confidence * 100).toFixed(1)}%`
      : null;

  // Geometry to scale original image onto fixed canvas
  const scale = Math.min(CANVAS / naturalSize.w, CANVAS / naturalSize.h);
  const dispW = naturalSize.w * scale;
  const dispH = naturalSize.h * scale;
  const offsetX = (CANVAS - dispW) / 2;
  const offsetY = (CANVAS - dispH) / 2;

  // Color by class (organism vs exuviae)
  const colorFor = (cls) => (cls === "organism" ? "red" : "blue");

  // Render YOLO boxes + labels on SVG overlay
  const renderBoxes = () => {
    const dets = resData?.detections || [];
    return dets.map((d, i) => {
      if (!d?.box || d.box.length !== 4) return null;

      const [x1, y1, x2, y2] = d.box;
      const sx = offsetX + x1 * scale;
      const sy = offsetY + y1 * scale;
      const w = (x2 - x1) * scale;
      const h = (y2 - y1) * scale;

      const color = colorFor(d.cls);
      const confTxt =
        typeof d.conf === "number" ? ` (${(d.conf * 100).toFixed(0)}%)` : "";
      const label = d.cls === "organism" ? "Organism" : "Exuviae";

      return (
        <g key={`box-${i}`}>
          <rect
            x={sx}
            y={sy}
            width={w}
            height={h}
            fill="none"
            stroke={color}
            strokeWidth="2"
          />
          <text
            x={sx + 4}
            y={Math.max(16, sy - 6)}
            fill={color}
            fontSize="16"
            fontWeight="bold"
          >
            {label}
            {confTxt}
          </text>
        </g>
      );
    });
  };

  // Only show masks if we have them and they are valid
  const maskIsVisible = (d) => {
    if (!d?.mask_png) return false;
    const q = (d?.quality || "").toLowerCase();
    return q !== "nomatch" && !q.startsWith("mask_error");
  };

  // Render instance segmentation masks (PNG base64 from backend)
  const renderMasks = () => {
    if (!showMask) return null;
    if (!resData?.use_seg || !resData?.detections?.length) return null;

    return resData.detections
      .filter((d) => maskIsVisible(d))
      .map((d, i) => (
        <img
          key={`mask-${i}`}
          src={`data:image/png;base64,${d.mask_png}`}
          alt="mask"
          style={{
            position: "absolute",
            top: offsetY,
            left: offsetX,
            width: dispW,
            height: dispH,
            pointerEvents: "none",
          }}
        />
      ));
  };

  return (
    <div className="app">
      <div className="container">
        <h1>MoultVision — Image Demo</h1>
        <p className="subtitle">
          Upload an image → YOLO + XGBoost (moulting stage)
          {resData?.use_seg ? " + Instance Segmentation" : ""}
        </p>

        {/* Top controls: image upload, taxon, predict, toggles */}
        <div
          className="row"
          style={{ alignItems: "center", gap: 12, flexWrap: "wrap" }}
        >
          <input
            id="image-input"
            className="input file"
            type="file"
            accept="image/*"
            onChange={onImageSelect}
          />

          <select
            className="input"
            value={taxonId}
            onChange={(e) => setTaxonId(e.target.value)}
          >
            <option value="">Select Taxon</option>
            {taxonOptions.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>

          <button
            onClick={onPredict}
            disabled={!image || taxonId === "" || imageLoading}
          >
            {imageLoading ? "Predicting..." : "Predict Image"}
          </button>

          {/* Toggle instance segmentation */}
          <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <input
              type="checkbox"
              checked={showMask}
              onChange={() => setShowMask(!showMask)}
            />
            Instance segmentation
          </label>

          <button
            className="clear-btn"
            onClick={onClear}
            style={{ marginLeft: "auto" }}
          >
            Clear
          </button>
        </div>

        {/* Error message (if any) */}
        {error && (
          <div className="loading-message" style={{ color: "#ff9aa2" }}>
            {error}
          </div>
        )}

        {/* Backend info + stage prediction */}
        {resData && !resData.error && (
          <div className="output" style={{ marginTop: 10 }}>
            <strong>Model:</strong> {resData.model} &nbsp;|&nbsp;
            <strong>Device:</strong> {resData.device} &nbsp;|&nbsp;
            <strong>Time:</strong> {resData.inference_ms} ms
            {stageLabel && (
              <>
                &nbsp;|&nbsp; <strong>Stage:</strong> {stageLabel}
                {stageConfPct && <> ({stageConfPct})</>}
              </>
            )}
            {resData?.seg_error && (
              <>
                &nbsp;|&nbsp;
                <span style={{ color: "#ff9aa2" }}>
                  Seg: {resData.seg_error}
                </span>
              </>
            )}
          </div>
        )}

        {/* Main visualization: original image + boxes + masks */}
        {imagePreview && (
          <div
            style={{
              position: "relative",
              width: CANVAS,
              height: CANVAS,
              marginTop: 15,
            }}
          >
            <img
              ref={imgRef}
              src={imagePreview}
              alt="Preview"
              style={{
                width: CANVAS,
                height: CANVAS,
                objectFit: "contain",
                background: "#111",
                borderRadius: 6,
              }}
              onLoad={(e) =>
                setNaturalSize({
                  w: e.currentTarget.naturalWidth,
                  h: e.currentTarget.naturalHeight,
                })
              }
            />

            {/* Segmentation masks under boxes for readability */}
            {renderMasks()}

            {/* Boxes overlay */}
            <svg
              width={CANVAS}
              height={CANVAS}
              style={{ position: "absolute", top: 0, left: 0 }}
            >
              {renderBoxes()}
            </svg>

            {/* Legend (Organism vs Exuviae colors) */}
            <div
              style={{
                position: "absolute",
                bottom: 8,
                left: 8,
                background: "rgba(0,0,0,0.5)",
                padding: "6px 8px",
                borderRadius: 6,
                fontSize: 12,
              }}
            >
              <span style={{ color: "red", fontWeight: 700 }}>■</span> Organism
              &nbsp;
              <span style={{ color: "blue", fontWeight: 700 }}>■</span> Exuviae
            </div>
          </div>
        )}

        {/* XGBoost errors (if any) */}
        {resData?.stage_error && (
          <div className="output" style={{ marginTop: 10, color: "#ff9aa2" }}>
            ⚠️ XGBoost stage error: {resData.stage_error}
          </div>
        )}

        {/* Generic backend errors (if any) */}
        {resData?.error && (
          <div className="output" style={{ marginTop: 10, color: "red" }}>
            ⚠️ {resData.error}
          </div>
        )}

        {/* Optional: dump debug features from backend (development only) */}
        {resData?.debug_features && (
          <pre
            style={{
              marginTop: 12,
              background: "#111",
              padding: 10,
              borderRadius: 6,
              overflowX: "auto",
            }}
          >
            {JSON.stringify(resData.debug_features, null, 2)}
          </pre>
        )}
      </div>
    </div>
  );
}

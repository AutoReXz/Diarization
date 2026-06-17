import { useState, useEffect } from 'react';
import { WHISPER_MODELS, LANGUAGES, API_BASE } from '../../constants/config';

export default function Sidebar({ config, onConfigChange }) {
  const [devices, setDevices] = useState(['cpu']);

  useEffect(() => {
    fetch(`${API_BASE}/config/devices`)
      .then((r) => r.json())
      .then((data) => {
        if (data.devices?.length) setDevices(data.devices);
      })
      .catch(() => {});
  }, []);

  const update = (key, value) => {
    onConfigChange({ ...config, [key]: value });
  };

  return (
    <aside className="sidebar">
      {/* Brand */}
      <div className="sidebar-header">
        <div className="sidebar-brand">
          <div className="sidebar-brand-icon">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9.348 14.652a3.75 3.75 0 0 1 0-5.304m5.304 0a3.75 3.75 0 0 1 0 5.304m-7.425 2.121a6.75 6.75 0 0 1 0-9.546m9.546 0a6.75 6.75 0 0 1 0 9.546M5.106 18.894c-3.808-3.807-3.808-9.98 0-13.788m13.788 0c3.808 3.808 3.808 9.981 0 13.789" />
            </svg>
          </div>
          <div className="sidebar-brand-text">
            <h2>Diarize AI</h2>
            <span>Self-hosted Pipeline</span>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="sidebar-content">
        {/* Configuration */}
        <div className="sidebar-section">
          <div className="sidebar-section-title">Configuration</div>
          <div className="form-group">
            <label className="form-label">HuggingFace Token</label>
            <input
              id="input-hf-token"
              type="password"
              className="form-input"
              placeholder="hf_..."
              value={config.hfToken}
              onChange={(e) => update('hfToken', e.target.value)}
            />
            <div className="form-hint">Required for pyannote diarization model</div>
          </div>
        </div>

        {/* Model */}
        <div className="sidebar-section">
          <div className="sidebar-section-title">Model</div>
          <div className="form-group">
            <label className="form-label">Whisper Model</label>
            <select
              id="select-model"
              className="form-select"
              value={config.modelName}
              onChange={(e) => update('modelName', e.target.value)}
            >
              {WHISPER_MODELS.map((m) => (
                <option key={m.value} value={m.value}>
                  {m.label} — {m.desc}
                </option>
              ))}
            </select>
          </div>
          <div className="form-group">
            <label className="form-label">Language</label>
            <select
              id="select-language"
              className="form-select"
              value={config.language}
              onChange={(e) => update('language', e.target.value)}
            >
              {LANGUAGES.map((l) => (
                <option key={l.value} value={l.value}>
                  {l.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Speakers */}
        <div className="sidebar-section">
          <div className="sidebar-section-title">Speaker Settings</div>
          <div className="form-group">
            <label className="form-label">Speaker Count</label>
            <div className="radio-group">
              {['auto', 'exact', 'range'].map((mode) => (
                <button
                  key={mode}
                  className={`radio-option ${config.speakerMode === mode ? 'active' : ''}`}
                  onClick={() => update('speakerMode', mode)}
                >
                  {mode === 'auto' ? 'Auto' : mode === 'exact' ? 'Exact' : 'Range'}
                </button>
              ))}
            </div>
          </div>

          {config.speakerMode === 'exact' && (
            <div className="form-group fade-in">
              <label className="form-label">Number of Speakers</label>
              <input
                id="input-num-speakers"
                type="number"
                className="form-input"
                min={1}
                max={20}
                value={config.numSpeakers}
                onChange={(e) => update('numSpeakers', parseInt(e.target.value, 10) || 2)}
              />
            </div>
          )}

          {config.speakerMode === 'range' && (
            <div className="number-row fade-in">
              <div className="form-group">
                <label className="form-label">Min</label>
                <input
                  id="input-min-speakers"
                  type="number"
                  className="form-input"
                  min={1}
                  max={20}
                  value={config.minSpeakers}
                  onChange={(e) => update('minSpeakers', parseInt(e.target.value, 10) || 2)}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Max</label>
                <input
                  id="input-max-speakers"
                  type="number"
                  className="form-input"
                  min={1}
                  max={20}
                  value={config.maxSpeakers}
                  onChange={(e) => update('maxSpeakers', parseInt(e.target.value, 10) || 5)}
                />
              </div>
            </div>
          )}
        </div>

        {/* Advanced */}
        <div className="sidebar-section">
          <div className="sidebar-section-title">Advanced</div>
          <div className="form-group">
            <label className="form-label">Compute Device</label>
            <select
              id="select-device"
              className="form-select"
              value={config.device}
              onChange={(e) => update('device', e.target.value)}
            >
              {devices.map((d) => (
                <option key={d} value={d}>
                  {d.toUpperCase()}
                </option>
              ))}
            </select>
            <div className="form-hint">GPU (CUDA/MPS) recommended for large models</div>
          </div>
          <div className="form-group">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <label className="form-label" style={{ marginBottom: 0 }}>Batch Size</label>
              <span className="slider-value">{config.batchSize}</span>
            </div>
            <input
              id="slider-batch-size"
              type="range"
              className="form-slider"
              min={1}
              max={32}
              value={config.batchSize}
              onChange={(e) => update('batchSize', parseInt(e.target.value, 10))}
            />
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="sidebar-footer">
        <div className="sidebar-footer-text">WhisperX + pyannote community model</div>
        <div className="tech-tags">
          <span className="tech-tag">WhisperX</span>
          <span className="tech-tag">pyannote</span>
          <span className="tech-tag">Open Source</span>
        </div>
      </div>
    </aside>
  );
}

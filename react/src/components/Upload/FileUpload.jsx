import { useState, useRef, useCallback } from 'react';
import { ACCEPTED_FORMATS } from '../../constants/config';
import { fmtFileSize } from '../../utils/format';

export default function FileUpload({ onFileSelect, uploadedFile, onRemove, disabled }) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);
  const audioUrl = uploadedFile ? URL.createObjectURL(uploadedFile) : null;

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragging(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      e.stopPropagation();
      setDragging(false);
      const file = e.dataTransfer?.files?.[0];
      if (file) onFileSelect(file);
    },
    [onFileSelect]
  );

  const handleClick = () => {
    inputRef.current?.click();
  };

  const handleChange = (e) => {
    const file = e.target.files?.[0];
    if (file) onFileSelect(file);
  };

  if (uploadedFile) {
    return (
      <div className="fade-in">
        <div className="file-info-bar">
          <div className="file-info-icon">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="m9 9 10.5-3m0 6.553v3.75a2.25 2.25 0 0 1-1.632 2.163l-1.32.377a1.803 1.803 0 1 1-.99-3.467l2.31-.66a2.25 2.25 0 0 0 1.632-2.163Zm0 0V4.125A2.25 2.25 0 0 0 17.378 2H6.622a2.25 2.25 0 0 0-2.122 1.5L2.25 9" />
            </svg>
          </div>
          <div className="file-info-details">
            <div className="file-info-name">{uploadedFile.name}</div>
            <div className="file-info-meta">{fmtFileSize(uploadedFile.size)}</div>
          </div>
          {!disabled && (
            <button className="file-remove-btn" onClick={onRemove} title="Remove file">
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
              </svg>
            </button>
          )}
        </div>
        {audioUrl && (
          <audio className="audio-player" controls src={audioUrl} />
        )}
      </div>
    );
  }

  return (
    <div>
      <div
        className={`upload-zone ${dragging ? 'dragging' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleClick}
        role="button"
        tabIndex={0}
        id="upload-zone"
      >
        <div className="upload-icon">
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5" />
          </svg>
        </div>
        <div className="upload-title">Drop your audio file here or click to browse</div>
        <div className="upload-desc">
          Drag and drop or click to select an audio/video file for transcription
        </div>
        <div className="upload-formats">
          {ACCEPTED_FORMATS.map((fmt) => (
            <span key={fmt} className="format-tag">
              {fmt}
            </span>
          ))}
        </div>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_FORMATS.map((f) => `.${f}`).join(',')}
        style={{ display: 'none' }}
        onChange={handleChange}
      />
    </div>
  );
}

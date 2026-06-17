import { useState } from 'react';
import Sidebar from './components/Layout/Sidebar';
import Header from './components/Layout/Header';
import FileUpload from './components/Upload/FileUpload';
import PipelineProgress from './components/Pipeline/PipelineProgress';
import ResultsDashboard from './components/Results/ResultsDashboard';
import usePipeline from './hooks/usePipeline';

const DEFAULT_CONFIG = {
  hfToken: '',
  modelName: 'large-v3',
  language: 'auto',
  speakerMode: 'auto',
  numSpeakers: 2,
  minSpeakers: 2,
  maxSpeakers: 5,
  device: 'cpu',
  batchSize: 16,
};

export default function App() {
  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const [uploadedFile, setUploadedFile] = useState(null);
  const { pipelineState, steps, results, error, run, reset } = usePipeline();

  const handleFileSelect = (file) => {
    setUploadedFile(file);
  };

  const handleRemoveFile = () => {
    setUploadedFile(null);
    reset();
  };

  const handleRun = () => {
    if (!uploadedFile || !config.hfToken) return;
    run(uploadedFile, config);
  };

  const isRunning = pipelineState === 'uploading' || pipelineState === 'running';
  const isDone = pipelineState === 'done' && results;

  return (
    <div className="app-layout">
      <Sidebar config={config} onConfigChange={setConfig} />

      <main className="main-content">
        <Header pipelineState={pipelineState} />

        <div className="content-area">
          {/* Upload Section */}
          <FileUpload
            onFileSelect={handleFileSelect}
            uploadedFile={uploadedFile}
            onRemove={handleRemoveFile}
            disabled={isRunning}
          />

          {/* Token Warning */}
          {uploadedFile && !config.hfToken && (
            <div
              className="fade-in"
              style={{
                marginTop: '1rem',
                padding: '0.85rem 1.15rem',
                borderRadius: 'var(--radius-md)',
                background: 'var(--accent-amber-soft)',
                border: '1px solid rgba(210, 153, 34, 0.25)',
                fontSize: '0.82rem',
                color: '#d29922',
              }}
            >
              A HuggingFace token is required for speaker diarization. Enter your token in the
              sidebar to proceed.
            </div>
          )}

          {/* Run Button */}
          {uploadedFile && !isDone && (
            <button
              className="run-button"
              onClick={handleRun}
              disabled={!config.hfToken || isRunning}
              id="btn-run-pipeline"
            >
              <span className="btn-content">
                {isRunning ? (
                  <>
                    <span className="spinner" />
                    Processing...
                  </>
                ) : (
                  <>
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.347a1.125 1.125 0 0 1 0 1.972l-11.54 6.347a1.125 1.125 0 0 1-1.667-.986V5.653Z" />
                    </svg>
                    Run Pipeline
                  </>
                )}
              </span>
            </button>
          )}

          {/* Error */}
          {error && (
            <div
              className="fade-in"
              style={{
                marginTop: '1rem',
                padding: '0.85rem 1.15rem',
                borderRadius: 'var(--radius-md)',
                background: 'var(--accent-red-soft)',
                border: '1px solid rgba(248, 81, 73, 0.25)',
                fontSize: '0.82rem',
                color: '#f85149',
              }}
            >
              Pipeline failed: {error}
            </div>
          )}

          {/* Pipeline Progress */}
          {isRunning && <PipelineProgress steps={steps} />}

          {/* Results */}
          {isDone && (
            <div style={{ marginTop: '2rem' }}>
              <ResultsDashboard
                segments={results.segments}
                timings={results.timings}
                sourceFile={uploadedFile?.name}
              />
            </div>
          )}

          {/* Empty State */}
          {!uploadedFile && pipelineState === 'idle' && (
            <div className="empty-state fade-in" style={{ marginTop: '2rem' }}>
              <div className="empty-icon">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                  strokeWidth={1.5}
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M12 18.75a6 6 0 0 0 6-6v-1.5m-6 7.5a6 6 0 0 1-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 0 1-3-3V4.5a3 3 0 1 1 6 0v8.25a3 3 0 0 1-3 3Z"
                  />
                </svg>
              </div>
              <div className="empty-title">Upload an audio file to get started</div>
              <div className="empty-desc">
                Drag and drop or click the upload area above to select an audio file. Configure
                your pipeline settings in the sidebar.
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

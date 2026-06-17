export default function Header({ pipelineState }) {
  const statusMap = {
    idle: { dot: 'idle', text: 'Ready' },
    running: { dot: 'processing', text: 'Processing...' },
    done: { dot: '', text: 'Complete' },
    error: { dot: 'error', text: 'Error' },
  };

  const status = statusMap[pipelineState] || statusMap.idle;

  return (
    <header className="app-header">
      <div className="header-inner">
        <div className="header-title">
          <h1>Speech-to-Text with Speaker Diarization</h1>
          <p>Upload audio to transcribe with automatic speaker identification</p>
        </div>
        <div className="header-status">
          <span className={`status-dot ${status.dot}`} />
          <span>{status.text}</span>
        </div>
      </div>
    </header>
  );
}

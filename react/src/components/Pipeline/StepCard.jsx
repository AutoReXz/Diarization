export default function StepCard({ number, label, title, detail, status }) {
  const stateClass = status || '';

  const renderIcon = () => {
    if (status === 'done') {
      return (
        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="none" viewBox="0 0 24 24" strokeWidth={2.5} stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" d="m4.5 12.75 6 6 9-13.5" />
        </svg>
      );
    }
    if (status === 'error') {
      return (
        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="none" viewBox="0 0 24 24" strokeWidth={2.5} stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
        </svg>
      );
    }
    return number;
  };

  return (
    <div className={`step-card ${stateClass}`}>
      <div className="step-number">{renderIcon()}</div>
      <div className="step-info">
        <div className="step-label">{label}</div>
        <div className="step-title">{title}</div>
        {detail && <div className="step-detail">{detail}</div>}
      </div>
    </div>
  );
}

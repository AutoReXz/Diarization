import { segmentsToText, segmentsToJson } from '../../utils/format';

function downloadFile(content, filename, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export default function ExportButtons({ segments, sourceFile }) {
  const baseName = sourceFile ? sourceFile.replace(/\.[^/.]+$/, '') : 'transcript';

  const handleDownloadTxt = () => {
    downloadFile(segmentsToText(segments), `${baseName}_diarized.txt`, 'text/plain');
  };

  const handleDownloadJson = () => {
    downloadFile(segmentsToJson(segments), `${baseName}_diarized.json`, 'application/json');
  };

  return (
    <div className="export-bar">
      <button className="export-btn" onClick={handleDownloadTxt}>
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5M16.5 12 12 16.5m0 0L7.5 12m4.5 4.5V3" />
        </svg>
        Download TXT
      </button>
      <button className="export-btn" onClick={handleDownloadJson}>
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5M16.5 12 12 16.5m0 0L7.5 12m4.5 4.5V3" />
        </svg>
        Download JSON
      </button>
    </div>
  );
}

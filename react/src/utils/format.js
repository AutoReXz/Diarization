/**
 * Format seconds to HH:MM:SS.mmm
 */
export function fmtTs(seconds) {
  const totalSecs = Math.floor(seconds);
  const hrs = Math.floor(totalSecs / 3600);
  const mins = Math.floor((totalSecs % 3600) / 60);
  const secs = totalSecs % 60;
  const millis = Math.round((seconds - totalSecs) * 1000);
  return `${String(hrs).padStart(2, '0')}:${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}.${String(millis).padStart(3, '0')}`;
}

/**
 * Human-readable duration from seconds.
 */
export function fmtDuration(seconds) {
  const totalSecs = Math.floor(seconds);
  const h = Math.floor(totalSecs / 3600);
  const m = Math.floor((totalSecs % 3600) / 60);
  const s = totalSecs % 60;
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

/**
 * Format file size in human readable format.
 */
export function fmtFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * Convert segments to plain text format for download.
 */
export function segmentsToText(segments) {
  return segments
    .map((seg) => {
      const start = fmtTs(seg.start);
      const end = fmtTs(seg.end);
      return `[${start} -> ${end}]  ${seg.speaker}:  ${seg.text}`;
    })
    .join('\n');
}

/**
 * Convert segments to pretty JSON for download.
 */
export function segmentsToJson(segments) {
  return JSON.stringify(segments, null, 2);
}

import { useMemo } from 'react';
import { fmtTs } from '../../utils/format';
import { speakerBadgeClass, speakerColor } from '../../utils/speakers';

export default function SpeakerTimeline({ segments }) {
  const totalDuration = useMemo(
    () => Math.max(...segments.map((s) => s.end), 0),
    [segments]
  );

  const speakers = useMemo(
    () => [...new Set(segments.map((s) => s.speaker))].sort(),
    [segments]
  );

  if (totalDuration <= 0) return null;

  return (
    <div>
      <div className="timeline-bar">
        {segments.map((seg, i) => {
          const widthPct = ((seg.end - seg.start) / totalDuration) * 100;
          const leftPct = (seg.start / totalDuration) * 100;
          const color = speakerColor(seg.speaker);
          return (
            <div
              key={i}
              className="timeline-seg"
              title={`${seg.speaker}: ${fmtTs(seg.start)} – ${fmtTs(seg.end)}`}
              style={{
                left: `${leftPct.toFixed(2)}%`,
                width: `${Math.max(widthPct, 0.3).toFixed(2)}%`,
                background: color,
                opacity: 0.7,
              }}
            />
          );
        })}
      </div>
      <div className="timeline-labels">
        <span>00:00:00</span>
        <span>{fmtTs(totalDuration)}</span>
      </div>
      <div className="timeline-legend">
        {speakers.map((spk) => (
          <span key={spk} className="timeline-legend-item">
            <span
              className="timeline-legend-swatch"
              style={{ background: speakerColor(spk) }}
            />
            {spk}
          </span>
        ))}
      </div>
    </div>
  );
}

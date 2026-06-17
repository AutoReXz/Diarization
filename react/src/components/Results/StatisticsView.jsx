import { useMemo } from 'react';
import MetricCard from '../ui/MetricCard';
import SpeakerBadge from '../ui/SpeakerBadge';
import SpeakerTimeline from './SpeakerTimeline';
import { fmtDuration } from '../../utils/format';

export default function StatisticsView({ segments, timings }) {
  const totalDuration = useMemo(
    () => Math.max(...segments.map((s) => s.end), 0),
    [segments]
  );

  const speakerData = useMemo(() => {
    const data = {};
    for (const seg of segments) {
      if (!data[seg.speaker]) {
        data[seg.speaker] = { segments: 0, words: 0, duration: 0 };
      }
      data[seg.speaker].segments += 1;
      data[seg.speaker].words += (seg.words || []).length;
      data[seg.speaker].duration += seg.end - seg.start;
    }
    return data;
  }, [segments]);

  const timingEntries = [
    { label: 'Transcription', key: 'transcription' },
    { label: 'Alignment', key: 'alignment' },
    { label: 'Diarization', key: 'diarization' },
    { label: 'Merge', key: 'merge' },
  ];

  return (
    <div className="fade-in">
      {/* Processing Time Breakdown */}
      <div className="stats-section">
        <div className="section-heading">Processing Time Breakdown</div>
        <div className="stats-grid-4 stagger">
          {timingEntries.map(({ label, key }) => {
            const val = timings[key] || 0;
            const pct = timings.total > 0 ? ((val / timings.total) * 100).toFixed(0) : 0;
            return (
              <MetricCard
                key={key}
                value={`${val.toFixed(1)}s`}
                label={`${label} (${pct}%)`}
              />
            );
          })}
        </div>
      </div>

      {/* Speaker Breakdown */}
      <div className="stats-section">
        <div className="section-heading">Speaker Breakdown</div>
        <div className="speaker-stats-grid stagger">
          {Object.entries(speakerData)
            .sort(([a], [b]) => a.localeCompare(b))
            .map(([spk, data]) => {
              const durPct =
                totalDuration > 0 ? ((data.duration / totalDuration) * 100).toFixed(1) : 0;
              return (
                <div key={spk} className="speaker-stat-card">
                  <div className="speaker-stat-header">
                    <SpeakerBadge
                      speaker={spk}
                      style={{ fontSize: '0.76rem', padding: '4px 14px' }}
                    />
                  </div>
                  <div className="stat-row">
                    <span className="stat-key">Segments</span>
                    <span className="stat-val">{data.segments}</span>
                  </div>
                  <div className="stat-row">
                    <span className="stat-key">Words</span>
                    <span className="stat-val">{data.words}</span>
                  </div>
                  <div className="stat-row">
                    <span className="stat-key">Duration</span>
                    <span className="stat-val">{fmtDuration(data.duration)}</span>
                  </div>
                  <div className="stat-row">
                    <span className="stat-key">Share</span>
                    <span className="stat-val">{durPct}%</span>
                  </div>
                </div>
              );
            })}
        </div>
      </div>

      {/* Timeline */}
      <div className="stats-section">
        <div className="section-heading">Speaker Timeline</div>
        <SpeakerTimeline segments={segments} />
      </div>
    </div>
  );
}

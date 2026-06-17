import { useState, useMemo } from 'react';
import TranscriptRow from './TranscriptRow';
import ExportButtons from '../ui/ExportButtons';

export default function TranscriptView({ segments, sourceFile }) {
  const [selectedSpeakers, setSelectedSpeakers] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');

  const speakers = useMemo(
    () => [...new Set(segments.map((s) => s.speaker))].sort(),
    [segments]
  );

  // Default: all speakers selected
  const activeSpeakers = selectedSpeakers || speakers;

  const filtered = useMemo(() => {
    return segments.filter((seg) => {
      const speakerMatch = activeSpeakers.includes(seg.speaker);
      const textMatch = !searchQuery || seg.text.toLowerCase().includes(searchQuery.toLowerCase());
      return speakerMatch && textMatch;
    });
  }, [segments, activeSpeakers, searchQuery]);

  const toggleSpeaker = (spk) => {
    const current = selectedSpeakers || [...speakers];
    if (current.includes(spk)) {
      const next = current.filter((s) => s !== spk);
      setSelectedSpeakers(next.length === 0 ? [...speakers] : next);
    } else {
      setSelectedSpeakers([...current, spk]);
    }
  };

  return (
    <div className="fade-in">
      <div className="transcript-toolbar">
        <div className="filter-group">
          <label className="form-label">Filter by speaker</label>
          <div className="filter-chips">
            {speakers.map((spk) => (
              <button
                key={spk}
                className={`filter-chip ${activeSpeakers.includes(spk) ? 'active' : ''}`}
                onClick={() => toggleSpeaker(spk)}
              >
                {spk}
              </button>
            ))}
          </div>
        </div>
        <div className="search-group">
          <label className="form-label">Search text</label>
          <input
            id="input-search-transcript"
            type="text"
            className="form-input"
            placeholder="Search..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
      </div>

      <div className="filter-count">
        Showing {filtered.length} of {segments.length} segments
      </div>

      <div className="transcript-container">
        {filtered.length > 0 ? (
          filtered.map((seg, i) => <TranscriptRow key={i} segment={seg} />)
        ) : (
          <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
            No segments match the current filter.
          </div>
        )}
      </div>

      <ExportButtons segments={segments} sourceFile={sourceFile} />
    </div>
  );
}

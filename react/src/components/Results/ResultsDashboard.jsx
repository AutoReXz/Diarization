import { useState, useMemo } from 'react';
import MetricCard from '../ui/MetricCard';
import Tabs from '../ui/Tabs';
import TranscriptView from './TranscriptView';
import JsonView from './JsonView';
import StatisticsView from './StatisticsView';
import { fmtDuration } from '../../utils/format';

const RESULT_TABS = [
  { id: 'transcript', label: 'Transcript' },
  { id: 'json', label: 'JSON' },
  { id: 'statistics', label: 'Statistics' },
];

export default function ResultsDashboard({ segments, timings, sourceFile }) {
  const [activeTab, setActiveTab] = useState('transcript');

  const speakers = useMemo(
    () => [...new Set(segments.map((s) => s.speaker))].sort(),
    [segments]
  );

  const totalDuration = useMemo(
    () => Math.max(...segments.map((s) => s.end), 0),
    [segments]
  );

  const totalWords = useMemo(
    () => segments.reduce((sum, seg) => sum + (seg.words || []).length, 0),
    [segments]
  );

  return (
    <div className="slide-up">
      {/* Summary Metrics */}
      <div className="metrics-grid stagger">
        <MetricCard value={String(segments.length)} label="Segments" />
        <MetricCard value={String(speakers.length)} label="Speakers" />
        <MetricCard value={String(totalWords)} label="Words" />
        <MetricCard value={fmtDuration(totalDuration)} label="Duration" />
        <MetricCard value={`${timings.total?.toFixed(1) || 0}s`} label="Processing" />
      </div>

      {/* Tabs */}
      <Tabs tabs={RESULT_TABS} activeTab={activeTab} onTabChange={setActiveTab} />

      {/* Tab Content */}
      {activeTab === 'transcript' && (
        <TranscriptView segments={segments} sourceFile={sourceFile} />
      )}
      {activeTab === 'json' && <JsonView segments={segments} />}
      {activeTab === 'statistics' && (
        <StatisticsView segments={segments} timings={timings} />
      )}
    </div>
  );
}

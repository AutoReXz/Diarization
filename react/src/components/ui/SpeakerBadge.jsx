import { speakerBadgeClass } from '../../utils/speakers';

export default function SpeakerBadge({ speaker, style }) {
  const cls = speakerBadgeClass(speaker);
  return (
    <span className={`speaker-badge ${cls}`} style={style}>
      {speaker}
    </span>
  );
}

import { fmtTs } from '../../utils/format';
import SpeakerBadge from '../ui/SpeakerBadge';

export default function TranscriptRow({ segment }) {
  return (
    <div className="transcript-row">
      <div className="transcript-ts">
        {fmtTs(segment.start)} → {fmtTs(segment.end)}
      </div>
      <div>
        <SpeakerBadge speaker={segment.speaker} />
      </div>
      <div className="transcript-text">{segment.text}</div>
    </div>
  );
}

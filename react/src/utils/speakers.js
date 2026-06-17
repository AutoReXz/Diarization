import { SPEAKER_COLORS } from '../constants/config';

/**
 * Get CSS class for a speaker label (e.g., "Speaker 1" -> "spk-1").
 */
export function speakerBadgeClass(speaker) {
  try {
    const num = parseInt(speaker.split(' ').pop(), 10);
    const idx = ((num - 1) % 8) + 1;
    return `spk-${idx}`;
  } catch {
    return 'spk-1';
  }
}

/**
 * Get hex color for a speaker label.
 */
export function speakerColor(speaker) {
  const cls = speakerBadgeClass(speaker);
  return SPEAKER_COLORS[cls] || SPEAKER_COLORS['spk-1'];
}

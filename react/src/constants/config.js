export const WHISPER_MODELS = [
  { value: 'tiny', label: 'Tiny', desc: 'Fastest, least accurate' },
  { value: 'base', label: 'Base', desc: 'Fast, basic accuracy' },
  { value: 'small', label: 'Small', desc: 'Good balance' },
  { value: 'medium', label: 'Medium', desc: 'High accuracy' },
  { value: 'large-v2', label: 'Large v2', desc: 'Very high accuracy' },
  { value: 'large-v3', label: 'Large v3', desc: 'Best accuracy' },
];

export const LANGUAGES = [
  { value: 'auto', label: 'Auto-detect' },
  { value: 'en', label: 'English' },
  { value: 'id', label: 'Indonesian' },
  { value: 'zh', label: 'Chinese' },
  { value: 'ja', label: 'Japanese' },
  { value: 'ko', label: 'Korean' },
  { value: 'de', label: 'German' },
  { value: 'fr', label: 'French' },
  { value: 'es', label: 'Spanish' },
  { value: 'pt', label: 'Portuguese' },
  { value: 'ru', label: 'Russian' },
  { value: 'ar', label: 'Arabic' },
  { value: 'hi', label: 'Hindi' },
  { value: 'it', label: 'Italian' },
  { value: 'nl', label: 'Dutch' },
  { value: 'pl', label: 'Polish' },
  { value: 'tr', label: 'Turkish' },
  { value: 'vi', label: 'Vietnamese' },
  { value: 'th', label: 'Thai' },
  { value: 'sv', label: 'Swedish' },
  { value: 'da', label: 'Danish' },
  { value: 'fi', label: 'Finnish' },
  { value: 'no', label: 'Norwegian' },
  { value: 'uk', label: 'Ukrainian' },
  { value: 'cs', label: 'Czech' },
];

export const SPEAKER_COLORS = {
  'spk-1': '#60a5fa',
  'spk-2': '#f472b6',
  'spk-3': '#34d399',
  'spk-4': '#fbbf24',
  'spk-5': '#a78bfa',
  'spk-6': '#fb923c',
  'spk-7': '#818cf8',
  'spk-8': '#c084fc',
};

export const ACCEPTED_FORMATS = [
  'wav', 'mp3', 'm4a', 'flac', 'ogg', 'wma', 'aac', 'mp4', 'webm', 'mkv',
];

export const ACCEPTED_MIME_TYPES = [
  'audio/wav', 'audio/mpeg', 'audio/mp4', 'audio/flac', 'audio/ogg',
  'audio/x-ms-wma', 'audio/aac', 'video/mp4', 'video/webm', 'video/x-matroska',
  'audio/x-wav', 'audio/x-m4a',
];

export const PIPELINE_STEPS = [
  { step: 1, label: 'Step 1 of 4', defaultTitle: 'Load ASR model & transcribe' },
  { step: 2, label: 'Step 2 of 4', defaultTitle: 'Run forced alignment' },
  { step: 3, label: 'Step 3 of 4', defaultTitle: 'Speaker diarization' },
  { step: 4, label: 'Step 4 of 4', defaultTitle: 'Merge results' },
];

export const API_BASE = '/api';

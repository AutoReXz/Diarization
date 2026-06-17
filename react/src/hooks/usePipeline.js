import { useState, useCallback, useRef } from 'react';
import { API_BASE } from '../constants/config';

/**
 * Hook managing the full pipeline lifecycle: upload → SSE progress → results.
 */
export default function usePipeline() {
  const [pipelineState, setPipelineState] = useState('idle'); // idle | uploading | running | done | error
  const [steps, setSteps] = useState([]);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [jobId, setJobId] = useState(null);
  const eventSourceRef = useRef(null);

  const updateStep = useCallback((stepData) => {
    setSteps((prev) => {
      const idx = prev.findIndex((s) => s.step === stepData.step);
      if (idx >= 0) {
        const copy = [...prev];
        copy[idx] = stepData;
        return copy;
      }
      return [...prev, stepData];
    });
  }, []);

  const run = useCallback(
    async (file, config) => {
      setError(null);
      setSteps([]);
      setResults(null);
      setPipelineState('uploading');

      try {
        // Step 0: Upload
        const uploadForm = new FormData();
        uploadForm.append('file', file);

        const uploadRes = await fetch(`${API_BASE}/upload`, {
          method: 'POST',
          body: uploadForm,
        });

        if (!uploadRes.ok) {
          const errData = await uploadRes.json().catch(() => ({}));
          throw new Error(errData.detail || `Upload failed: ${uploadRes.status}`);
        }

        const { job_id } = await uploadRes.json();
        setJobId(job_id);
        setPipelineState('running');

        // Step 1-4: SSE pipeline
        const pipelineForm = new FormData();
        pipelineForm.append('job_id', job_id);
        pipelineForm.append('model_name', config.modelName);
        pipelineForm.append('language', config.language);
        pipelineForm.append('device', config.device);
        pipelineForm.append('batch_size', String(config.batchSize));
        pipelineForm.append('hf_token', config.hfToken);

        if (config.speakerMode === 'exact' && config.numSpeakers) {
          pipelineForm.append('num_speakers', String(config.numSpeakers));
        }
        if (config.speakerMode === 'range') {
          if (config.minSpeakers) pipelineForm.append('min_speakers', String(config.minSpeakers));
          if (config.maxSpeakers) pipelineForm.append('max_speakers', String(config.maxSpeakers));
        }

        // Use fetch with ReadableStream for SSE from POST
        const response = await fetch(`${API_BASE}/pipeline/run`, {
          method: 'POST',
          body: pipelineForm,
        });

        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          throw new Error(errData.detail || `Pipeline failed: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          let currentEvent = null;
          for (const line of lines) {
            if (line.startsWith('event:')) {
              currentEvent = line.slice(6).trim();
            } else if (line.startsWith('data:')) {
              const dataStr = line.slice(5).trim();
              if (!dataStr) continue;
              try {
                const data = JSON.parse(dataStr);
                if (currentEvent === 'step') {
                  updateStep(data);
                } else if (currentEvent === 'complete') {
                  setResults(data);
                  setPipelineState('done');
                }
              } catch {
                // skip malformed JSON
              }
              currentEvent = null;
            }
          }
        }

        // If we exited the loop without receiving 'complete'
        if (pipelineState !== 'done' && !results) {
          // Check if results were set
        }
      } catch (err) {
        setError(err.message);
        setPipelineState('error');
      }
    },
    [updateStep]
  );

  const reset = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    setPipelineState('idle');
    setSteps([]);
    setResults(null);
    setError(null);
    setJobId(null);
  }, []);

  return {
    pipelineState,
    steps,
    results,
    error,
    jobId,
    run,
    reset,
  };
}

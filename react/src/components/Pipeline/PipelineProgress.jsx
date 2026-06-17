import StepCard from './StepCard';
import { PIPELINE_STEPS } from '../../constants/config';

export default function PipelineProgress({ steps }) {
  return (
    <div className="pipeline-progress fade-in">
      <div className="pipeline-header">
        <div className="pipeline-title">Pipeline Progress</div>
      </div>
      <div className="stagger">
        {PIPELINE_STEPS.map((pStep) => {
          const stepState = steps.find((s) => s.step === pStep.step);
          return (
            <StepCard
              key={pStep.step}
              number={pStep.step}
              label={pStep.label}
              title={stepState?.title || pStep.defaultTitle}
              detail={stepState?.detail || ''}
              status={stepState?.status || ''}
            />
          );
        })}
      </div>
    </div>
  );
}

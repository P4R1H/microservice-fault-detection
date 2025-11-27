import { motion, AnimatePresence } from 'framer-motion';
import { useState } from 'react';
import {
  Activity,
  FileText,
  Network,
  Cpu,
  Layers,
  Brain,
  Zap,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';

const steps = [
  {
    id: 0,
    title: 'Metrics Input',
    subtitle: 'Time-Series Data',
    icon: Activity,
    color: 'blue',
    content: {
      description: 'Real-time telemetry from each microservice including CPU utilization, memory usage, request latency, and error rates.',
      details: [
        '64 features per service',
        'T timesteps of history',
        'Normalized & windowed',
      ],
      tech: 'Per-service metric tensors: S × T × 64',
    },
  },
  {
    id: 1,
    title: 'Logs Input',
    subtitle: 'TF-IDF Encoding',
    icon: FileText,
    color: 'blue',
    content: {
      description: 'Error messages, warnings, and system events encoded using learnable TF-IDF weights that adapt to RCA-specific patterns.',
      details: [
        'Template-based encoding',
        'Learnable importance weights',
        'Error pattern detection',
      ],
      tech: '64-dimensional log embeddings per service',
    },
  },
  {
    id: 2,
    title: 'Traces Input',
    subtitle: 'Call Graph Features',
    icon: Network,
    color: 'blue',
    content: {
      description: 'Request flow patterns and service dependencies extracted from distributed traces, capturing latency propagation.',
      details: [
        'Inter-service call patterns',
        'Latency & error propagation',
        'Dependency graph structure',
      ],
      tech: '32 trace features × T timesteps',
    },
  },
  {
    id: 3,
    title: 'TCN Encoder',
    subtitle: 'Temporal Pattern Extraction',
    icon: Cpu,
    color: 'blue',
    content: {
      description: 'Depthwise separable Temporal Convolutional Networks extract patterns from each modality with 8× fewer parameters than standard convolutions.',
      details: [
        '2 temporal blocks per modality',
        'Dilated convolutions (d=1,2)',
        'BatchNorm + GELU + 35% Dropout',
      ],
      tech: 'Output: 64-dim embedding per modality',
    },
  },
  {
    id: 4,
    title: 'Gated Fusion',
    subtitle: 'Learned Modality Weighting',
    icon: Layers,
    color: 'blue',
    content: {
      description: 'Instead of fixed concatenation, learned gates determine how much each modality contributes to each prediction.',
      details: [
        'g = σ(W·[metrics, logs, traces])',
        'Per-case modality importance',
        'Interpretable gate values',
      ],
      tech: 'fused = g_m·M + g_l·L + g_t·T → 128d',
    },
  },
  {
    id: 5,
    title: 'Cross-Service Attention',
    subtitle: 'Multi-Head Self-Attention',
    icon: Brain,
    color: 'blue',
    content: {
      description: 'Services attend to each other to identify which service behaviors correlate with the failure, enabling cross-service reasoning.',
      details: [
        '2 attention layers',
        '4 attention heads',
        'Residual connections',
      ],
      tech: 'Attention(Q,K,V) with 128-dim queries',
    },
  },
  {
    id: 6,
    title: 'PCMCI Causal Injection',
    subtitle: 'Causal Discovery Biasing',
    icon: () => (
      <svg className="w-6 h-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="12" r="3" />
        <path d="M12 2v4m0 12v4M2 12h4m12 0h4" />
        <path d="M4.93 4.93l2.83 2.83m8.48 8.48l2.83 2.83M4.93 19.07l2.83-2.83m8.48-8.48l2.83-2.83" />
      </svg>
    ),
    color: 'purple',
    content: {
      description: 'PCMCI causal discovery weights are injected into attention scores, biasing the model toward causally-related services.',
      details: [
        'Pre-computed causal matrix C',
        'Attention += λ·C (λ=0.3)',
        'Grounds predictions in causality',
      ],
      tech: 'softmax(QK^T/√d + λ·C)·V',
    },
  },
  {
    id: 7,
    title: 'Root Cause Output',
    subtitle: 'Service Ranking',
    icon: Zap,
    color: 'blue',
    content: {
      description: 'Final MLP scoring head produces confidence scores for each service, ranked to identify the most likely root cause.',
      details: [
        'MLP: 128 → 64 → 1',
        'Softmax over services',
        'Top-k ranking output',
      ],
      tech: '88.9% AC@1 accuracy | ~3ms inference',
    },
  },
];

export function Architecture() {
  const [currentStep, setCurrentStep] = useState(0);
  
  const nextStep = () => setCurrentStep((prev) => (prev + 1) % steps.length);
  const prevStep = () => setCurrentStep((prev) => (prev - 1 + steps.length) % steps.length);
  
  const step = steps[currentStep];
  const Icon = step.icon;

  return (
    <section id="architecture" className="relative py-32 overflow-hidden">
      {/* Subtle background */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(59,130,246,0.05)_0%,transparent_70%)]" />
      
      <div className="relative z-10 max-w-5xl mx-auto px-6">
        {/* Section header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-12"
        >
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
            How It <span className="text-gradient">Works</span>
          </h2>
          <p className="text-lg text-zinc-400 max-w-2xl mx-auto">
            Click through each stage of our multimodal architecture
          </p>
        </motion.div>

        {/* Progress indicators */}
        <div className="flex items-center justify-center gap-2 mb-8">
          {steps.map((s, i) => (
            <button
              key={s.id}
              onClick={() => setCurrentStep(i)}
              className={`h-1.5 rounded-full transition-all duration-300 ${
                i === currentStep 
                  ? 'w-8 bg-blue-500' 
                  : i < currentStep 
                    ? 'w-3 bg-blue-500/50' 
                    : 'w-3 bg-zinc-700'
              }`}
            />
          ))}
        </div>

        {/* Main bento card */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="relative"
        >
          {/* Navigation buttons */}
          <button
            onClick={prevStep}
            className="absolute left-0 top-1/2 -translate-y-1/2 -translate-x-4 md:-translate-x-12 z-20 w-10 h-10 rounded-full bg-zinc-800/80 border border-zinc-700 flex items-center justify-center text-zinc-400 hover:text-white hover:border-blue-500/50 transition-all"
          >
            <ChevronLeft className="w-5 h-5" />
          </button>
          <button
            onClick={nextStep}
            className="absolute right-0 top-1/2 -translate-y-1/2 translate-x-4 md:translate-x-12 z-20 w-10 h-10 rounded-full bg-zinc-800/80 border border-zinc-700 flex items-center justify-center text-zinc-400 hover:text-white hover:border-blue-500/50 transition-all"
          >
            <ChevronRight className="w-5 h-5" />
          </button>

          <div className="rounded-2xl bg-zinc-900/50 border border-zinc-800/50 overflow-hidden">
            {/* Step header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-800/50 bg-zinc-900/30">
              <div className="flex items-center gap-3">
                <span className="text-xs font-mono text-zinc-500">STEP {currentStep + 1} OF {steps.length}</span>
              </div>
              <div className="flex items-center gap-2">
                <span className={`text-xs font-medium px-2 py-1 rounded-full ${
                  currentStep < 3 ? 'bg-blue-500/20 text-blue-400' :
                  currentStep < 7 ? 'bg-purple-500/20 text-purple-400' :
                  'bg-blue-500/20 text-blue-400'
                }`}>
                  {currentStep < 3 ? 'Input' : currentStep < 7 ? 'Processing' : 'Output'}
                </span>
              </div>
            </div>

            {/* Step content */}
            <div className="p-8 min-h-[400px]">
              <AnimatePresence mode="wait">
                <motion.div
                  key={currentStep}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.3 }}
                  className="h-full"
                >
                  <div className="flex flex-col md:flex-row gap-8">
                    {/* Left: Icon and title */}
                    <div className="md:w-1/3">
                      <div className={`w-20 h-20 rounded-2xl flex items-center justify-center mb-6 ${
                        step.color === 'purple' ? 'bg-purple-500/20' : 'bg-blue-500/20'
                      }`}>
                        <div className={step.color === 'purple' ? 'text-purple-400' : 'text-blue-400'}>
                          {typeof Icon === 'function' && Icon.length === 0 ? <Icon /> : <Icon className="w-10 h-10" />}
                        </div>
                      </div>
                      <h3 className="text-2xl font-bold text-white mb-2">{step.title}</h3>
                      <p className={`text-lg ${step.color === 'purple' ? 'text-purple-400' : 'text-blue-400'}`}>
                        {step.subtitle}
                      </p>
                    </div>

                    {/* Right: Content */}
                    <div className="md:w-2/3 space-y-6">
                      <p className="text-zinc-300 text-lg leading-relaxed">
                        {step.content.description}
                      </p>

                      <div className="space-y-3">
                        {step.content.details.map((detail, i) => (
                          <motion.div
                            key={i}
                            initial={{ opacity: 0, x: 10 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: i * 0.1 }}
                            className="flex items-center gap-3"
                          >
                            <div className={`w-1.5 h-1.5 rounded-full ${
                              step.color === 'purple' ? 'bg-purple-400' : 'bg-blue-400'
                            }`} />
                            <span className="text-zinc-400">{detail}</span>
                          </motion.div>
                        ))}
                      </div>

                      <div className={`p-4 rounded-xl ${
                        step.color === 'purple' ? 'bg-purple-500/10 border border-purple-500/20' : 'bg-blue-500/10 border border-blue-500/20'
                      }`}>
                        <p className="text-sm font-mono text-zinc-300">
                          {step.content.tech}
                        </p>
                      </div>
                    </div>
                  </div>
                </motion.div>
              </AnimatePresence>
            </div>

            {/* Bottom navigation */}
            <div className="flex items-center justify-between px-8 py-4 border-t border-zinc-800/50 bg-zinc-900/30">
              <button
                onClick={prevStep}
                className="flex items-center gap-2 text-zinc-400 hover:text-white transition-colors"
              >
                <ChevronLeft className="w-4 h-4" />
                <span className="text-sm">Previous</span>
              </button>
              
              <div className="text-sm text-zinc-500">
                {steps[currentStep].title}
              </div>
              
              <button
                onClick={nextStep}
                className="flex items-center gap-2 text-zinc-400 hover:text-white transition-colors"
              >
                <span className="text-sm">Next</span>
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        </motion.div>

        {/* Model specs */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="mt-12"
        >
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { value: '324K', label: 'Parameters' },
              { value: '~3ms', label: 'Inference' },
              { value: '88.9%', label: 'AC@1' },
              { value: '272×', label: 'Faster' },
            ].map((stat) => (
              <div key={stat.label} className="p-4 rounded-xl bg-zinc-900/30 border border-zinc-800/30 text-center">
                <p className="text-xl font-bold text-white">{stat.value}</p>
                <p className="text-xs text-zinc-500 mt-1">{stat.label}</p>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  );
}

import { motion, AnimatePresence } from 'framer-motion';
import { useState } from 'react';
import {
  Activity, FileText, Network, Layers, GitMerge, Brain, Cpu, Zap,
  Sparkles, Target, Hash, Timer, Gauge, Box, Combine, GitBranch,
  MessageSquare, Workflow, ChevronDown, Database
} from 'lucide-react';

// --- Types & Data ---
type StageId = 'input' | 'tcn' | 'fusion' | 'attention' | 'llm' | 'output';

interface StageInfo {
  title: string;
  subtitle: string;
  icon: React.ElementType;
  color: string;
  stats: Array<{
    icon: React.ElementType;
    value: string;
    unit?: string;
    label: string;
    color: string;
  }>;
}

const stageDetails: Record<StageId, StageInfo> = {
  input: {
    title: 'Multimodal Input',
    subtitle: 'Raw Data Ingestion',
    icon: Database,
    color: 'blue',
    stats: [
      { icon: Activity, value: '64', label: 'Metric Feat', color: 'blue' },
      { icon: FileText, value: '32', label: 'Log Templates', color: 'emerald' },
      { icon: Network, value: '32', label: 'Trace Feat', color: 'purple' },
      { icon: Timer, value: '60', label: 'Time Steps', color: 'amber' },
    ],
  },
  tcn: {
    title: 'TCN Encoders',
    subtitle: 'Temporal Convolution',
    icon: Layers,
    color: 'amber',
    stats: [
      { icon: Layers, value: '2', label: 'Blocks', color: 'amber' },
      { icon: Hash, value: '[1,2]', label: 'Dilation', color: 'blue' },
      { icon: Box, value: '3', label: 'Kernel', color: 'purple' },
      { icon: Gauge, value: '64', unit: 'd', label: 'Embed Dim', color: 'emerald' },
    ],
  },
  fusion: {
    title: 'Gated Fusion',
    subtitle: 'Modality Merging',
    icon: GitMerge,
    color: 'pink',
    stats: [
      { icon: Combine, value: '3', label: 'Modalities', color: 'pink' },
      { icon: GitMerge, value: 'σ(W)', label: 'Gate Func', color: 'blue' },
      { icon: Gauge, value: '128', unit: 'd', label: 'Fused Dim', color: 'purple' },
      { icon: Workflow, value: 'Auto', label: 'Adaptive', color: 'amber' },
    ],
  },
  attention: {
    title: 'Cross-Service Attention',
    subtitle: 'Causal Discovery',
    icon: Brain,
    color: 'cyan',
    stats: [
      { icon: Brain, value: '4', label: 'Heads', color: 'cyan' },
      { icon: Layers, value: '2', label: 'Tx Layers', color: 'blue' },
      { icon: GitBranch, value: '0.3', label: 'Causal λ', color: 'purple' },
      { icon: Network, value: 'DAG', label: 'Graph', color: 'emerald' },
    ],
  },
  llm: {
    title: 'LLM Causal Prior',
    subtitle: 'Gemini Reasoning',
    icon: Sparkles,
    color: 'purple',
    stats: [
      { icon: Sparkles, value: 'Gemini', label: 'Model', color: 'purple' },
      { icon: MessageSquare, value: 'CoT', label: 'Reasoning', color: 'blue' },
      { icon: Brain, value: 'Hybrid', label: 'Arch', color: 'amber' },
      { icon: Target, value: '+3.2', unit: '%', label: 'Lift', color: 'pink' },
    ],
  },
  output: {
    title: 'Root Cause Prediction',
    subtitle: 'Final Inference',
    icon: Target,
    color: 'emerald',
    stats: [
      { icon: Target, value: '88.9', unit: '%', label: 'Accuracy', color: 'emerald' },
      { icon: Zap, value: '3.3', unit: 'ms', label: 'Latency', color: 'amber' },
      { icon: Cpu, value: '324', unit: 'K', label: 'Params', color: 'blue' },
      { icon: Gauge, value: '0.94', label: 'MRR', color: 'purple' },
    ],
  },
};

const stages: StageId[] = ['input', 'tcn', 'fusion', 'attention', 'llm', 'output'];

export function Architecture() {
  const [activeId, setActiveId] = useState<StageId>('input');

  const activeInfo = stageDetails[activeId];

  return (
    <section id="architecture" className="relative py-32 overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(59,130,246,0.03)_0%,transparent_70%)]" />
      
      <div className="relative z-10 max-w-6xl mx-auto px-6">
        {/* Section header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-12"
        >
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
            Our <span className="text-gradient">Architecture</span>
          </h2>
          <p className="text-lg text-zinc-400 max-w-2xl mx-auto">
            Multimodal deep learning with causal reasoning and LLM enhancement
          </p>
        </motion.div>

        {/* Two-column layout */}
        <motion.div 
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="grid grid-cols-1 lg:grid-cols-5 gap-6"
        >
          {/* Left: Stage List */}
          <div className="lg:col-span-2 space-y-2">
            {stages.map((stage) => {
              const isActive = activeId === stage;
              const info = stageDetails[stage];
              const StageIcon = info.icon;

              return (
                <motion.div
                  key={stage}
                  onClick={() => setActiveId(stage)}
                  whileHover={{ x: 4 }}
                  className={`relative overflow-hidden cursor-pointer rounded-xl border transition-all duration-200 ${
                    isActive 
                      ? 'bg-zinc-900/80 border-zinc-700' 
                      : 'bg-zinc-900/40 border-zinc-800/50 hover:bg-zinc-900/60 hover:border-zinc-700/50'
                  }`}
                >
                  {/* Sidebar Indicator */}
                  <div className={`absolute left-0 top-0 bottom-0 w-1 rounded-l-xl transition-all duration-200 ${
                    isActive ? `bg-${info.color}-500` : `bg-${info.color}-500/20`
                  }`} />

                  <div className="p-4 flex items-center gap-3">
                    <div className={`w-9 h-9 rounded-lg flex items-center justify-center bg-${info.color}-500/20`}>
                      <StageIcon className={`w-4 h-4 text-${info.color}-400`} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <h3 className={`font-medium text-sm transition-colors ${isActive ? 'text-white' : 'text-zinc-400'}`}>
                        {info.title}
                      </h3>
                      <p className="text-xs text-zinc-600 truncate">{info.subtitle}</p>
                    </div>
                    <ChevronDown className={`w-4 h-4 transition-transform duration-200 ${
                      isActive ? 'text-zinc-400 -rotate-90' : 'text-zinc-600 -rotate-90'
                    }`} />
                  </div>
                </motion.div>
              );
            })}
          </div>

          {/* Right: Detail Card */}
          <div className="lg:col-span-3">
            <AnimatePresence mode="wait">
              <motion.div
                key={activeId}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.2 }}
                className="bg-zinc-900/60 border border-zinc-800/50 rounded-2xl p-6 h-full"
              >
                {/* Header */}
                <div className="flex items-center gap-4 mb-6">
                  <div className={`w-12 h-12 rounded-xl flex items-center justify-center bg-${activeInfo.color}-500/20`}>
                    <activeInfo.icon className={`w-6 h-6 text-${activeInfo.color}-400`} />
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold text-white">{activeInfo.title}</h3>
                    <p className="text-sm text-zinc-500">{activeInfo.subtitle}</p>
                  </div>
                </div>

                {/* Description */}
                <p className="text-zinc-400 text-sm leading-relaxed mb-6 border-l-2 border-zinc-700 pl-4">
                  Processing logic for <span className="text-white font-medium">{activeInfo.subtitle}</span>. 
                  Optimized for high-dimensional throughput with minimal computational overhead.
                </p>

                {/* Stats Grid */}
                <div className="grid grid-cols-2 gap-3">
                  {activeInfo.stats.map((stat, i) => (
                    <motion.div 
                      key={i}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: i * 0.05 }}
                      className="bg-zinc-950/50 border border-zinc-800/50 p-4 rounded-xl hover:border-zinc-700/50 transition-colors"
                    >
                      <div className="flex items-center gap-2 mb-2">
                        <stat.icon className={`w-4 h-4 text-${stat.color}-400`} />
                        <span className="text-[10px] uppercase tracking-wider text-zinc-500 font-medium">{stat.label}</span>
                      </div>
                      <div className="text-2xl font-bold text-white">
                        {stat.value}
                        {stat.unit && <span className="text-base text-zinc-500 ml-0.5">{stat.unit}</span>}
                      </div>
                    </motion.div>
                  ))}
                </div>
              </motion.div>
            </AnimatePresence>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
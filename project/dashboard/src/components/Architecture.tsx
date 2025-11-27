import { motion, AnimatePresence } from 'framer-motion';
import { useState } from 'react';
import {
  Activity, FileText, Network, Layers, GitMerge, Brain, Cpu, Zap,
  Sparkles, Target, Hash, Timer, Gauge, Box, Combine, GitBranch,
  MessageSquare, Workflow, Database
} from 'lucide-react';

// --- Types & Data ---
type StageId = 'input' | 'tcn' | 'fusion' | 'attention' | 'llm' | 'output';

interface StageInfo {
  title: string;
  subtitle: string;
  icon: React.ElementType;
  colorClass: {
    bg: string;
    bgLight: string;
    text: string;
    border: string;
  };
  stats: Array<{
    icon: React.ElementType;
    value: string;
    unit?: string;
    label: string;
    textColor: string;
  }>;
}

// Define color classes explicitly for Tailwind JIT
const colorClasses = {
  blue: { bg: 'bg-blue-500', bgLight: 'bg-blue-500/20', text: 'text-blue-400', border: 'border-blue-500' },
  amber: { bg: 'bg-amber-500', bgLight: 'bg-amber-500/20', text: 'text-amber-400', border: 'border-amber-500' },
  pink: { bg: 'bg-pink-500', bgLight: 'bg-pink-500/20', text: 'text-pink-400', border: 'border-pink-500' },
  cyan: { bg: 'bg-cyan-500', bgLight: 'bg-cyan-500/20', text: 'text-cyan-400', border: 'border-cyan-500' },
  purple: { bg: 'bg-purple-500', bgLight: 'bg-purple-500/20', text: 'text-purple-400', border: 'border-purple-500' },
  emerald: { bg: 'bg-emerald-500', bgLight: 'bg-emerald-500/20', text: 'text-emerald-400', border: 'border-emerald-500' },
};

const stageDetails: Record<StageId, StageInfo> = {
  input: {
    title: 'Multimodal Input',
    subtitle: 'Raw Data Ingestion',
    icon: Database,
    colorClass: colorClasses.blue,
    stats: [
      { icon: Activity, value: '64', label: 'Metric Feat', textColor: 'text-blue-400' },
      { icon: FileText, value: '32', label: 'Log Templates', textColor: 'text-emerald-400' },
      { icon: Network, value: '32', label: 'Trace Feat', textColor: 'text-purple-400' },
      { icon: Timer, value: '60', label: 'Time Steps', textColor: 'text-amber-400' },
    ],
  },
  tcn: {
    title: 'TCN Encoders',
    subtitle: 'Temporal Convolution',
    icon: Layers,
    colorClass: colorClasses.amber,
    stats: [
      { icon: Layers, value: '2', label: 'Blocks', textColor: 'text-amber-400' },
      { icon: Hash, value: '[1,2]', label: 'Dilation', textColor: 'text-blue-400' },
      { icon: Box, value: '3', label: 'Kernel', textColor: 'text-purple-400' },
      { icon: Gauge, value: '64', unit: 'd', label: 'Embed Dim', textColor: 'text-emerald-400' },
    ],
  },
  fusion: {
    title: 'Gated Fusion',
    subtitle: 'Modality Merging',
    icon: GitMerge,
    colorClass: colorClasses.pink,
    stats: [
      { icon: Combine, value: '3', label: 'Modalities', textColor: 'text-pink-400' },
      { icon: GitMerge, value: 'σ(W)', label: 'Gate Func', textColor: 'text-blue-400' },
      { icon: Gauge, value: '128', unit: 'd', label: 'Fused Dim', textColor: 'text-purple-400' },
      { icon: Workflow, value: 'Auto', label: 'Adaptive', textColor: 'text-amber-400' },
    ],
  },
  attention: {
    title: 'Cross-Service Attention',
    subtitle: 'Causal Discovery',
    icon: Brain,
    colorClass: colorClasses.cyan,
    stats: [
      { icon: Brain, value: '4', label: 'Heads', textColor: 'text-cyan-400' },
      { icon: Layers, value: '2', label: 'Tx Layers', textColor: 'text-blue-400' },
      { icon: GitBranch, value: '0.3', label: 'Causal λ', textColor: 'text-purple-400' },
      { icon: Network, value: 'DAG', label: 'Graph', textColor: 'text-emerald-400' },
    ],
  },
  llm: {
    title: 'LLM Causal Prior',
    subtitle: 'Gemini Reasoning',
    icon: Sparkles,
    colorClass: colorClasses.purple,
    stats: [
      { icon: Sparkles, value: 'Gemini', label: 'Model', textColor: 'text-purple-400' },
      { icon: MessageSquare, value: 'CoT', label: 'Reasoning', textColor: 'text-blue-400' },
      { icon: Brain, value: 'Hybrid', label: 'Arch', textColor: 'text-amber-400' },
      { icon: Target, value: '+3.2', unit: '%', label: 'Lift', textColor: 'text-pink-400' },
    ],
  },
  output: {
    title: 'Root Cause Prediction',
    subtitle: 'Final Inference',
    icon: Target,
    colorClass: colorClasses.emerald,
    stats: [
      { icon: Target, value: '88.9', unit: '%', label: 'Accuracy', textColor: 'text-emerald-400' },
      { icon: Zap, value: '3.3', unit: 'ms', label: 'Latency', textColor: 'text-amber-400' },
      { icon: Cpu, value: '324', unit: 'K', label: 'Params', textColor: 'text-blue-400' },
      { icon: Gauge, value: '0.94', label: 'MRR', textColor: 'text-purple-400' },
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
                    isActive ? info.colorClass.bg : `${info.colorClass.bgLight}`
                  }`} />

                  <div className="p-4 flex items-center gap-3">
                    <div className={`w-9 h-9 rounded-lg flex items-center justify-center ${info.colorClass.bgLight}`}>
                      <StageIcon className={`w-4 h-4 ${info.colorClass.text}`} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <h3 className={`font-medium text-sm transition-colors ${isActive ? 'text-white' : 'text-zinc-400'}`}>
                        {info.title}
                      </h3>
                      <p className="text-xs text-zinc-600 truncate">{info.subtitle}</p>
                    </div>
                    <svg className={`w-4 h-4 transition-colors ${isActive ? 'text-zinc-400' : 'text-zinc-600'}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
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
                  <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${activeInfo.colorClass.bgLight}`}>
                    <activeInfo.icon className={`w-6 h-6 ${activeInfo.colorClass.text}`} />
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
                        <stat.icon className={`w-4 h-4 ${stat.textColor}`} />
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
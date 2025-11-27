import { motion } from 'framer-motion';
import {
  Activity,
  FileText,
  Network,
  Layers,
  GitMerge,
  Brain,
  Target,
  ChevronRight,
} from 'lucide-react';

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.1 },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 },
};

export function Architecture() {
  return (
    <section id="architecture" className="relative py-32 overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(59,130,246,0.03)_0%,transparent_70%)]" />
      
      <div className="relative z-10 max-w-7xl mx-auto px-6">
        {/* Section header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
            How It <span className="text-gradient">Works</span>
          </h2>
          <p className="text-lg text-zinc-400 max-w-2xl mx-auto">
            A lightweight multimodal pipeline that fuses metrics, logs, and traces
          </p>
        </motion.div>

        {/* Bento Grid - matching the style of BentoStats */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4"
        >
          {/* Large card: Pipeline Overview */}
          <motion.div
            variants={itemVariants}
            className="bento-card p-6 lg:col-span-2 lg:row-span-2"
          >
            <div className="flex items-center gap-2 mb-6">
              <Layers className="w-5 h-5 text-blue-400" />
              <span className="text-blue-400 text-sm font-semibold uppercase tracking-wide">Pipeline</span>
            </div>
            
            {/* Visual pipeline flow */}
            <div className="space-y-4">
              {/* Input stage */}
              <div className="flex items-center gap-3">
                <div className="flex gap-2">
                  <div className="w-10 h-10 rounded-xl bg-blue-500/20 flex items-center justify-center">
                    <Activity className="w-5 h-5 text-blue-400" />
                  </div>
                  <div className="w-10 h-10 rounded-xl bg-emerald-500/20 flex items-center justify-center">
                    <FileText className="w-5 h-5 text-emerald-400" />
                  </div>
                  <div className="w-10 h-10 rounded-xl bg-purple-500/20 flex items-center justify-center">
                    <Network className="w-5 h-5 text-purple-400" />
                  </div>
                </div>
                <ChevronRight className="w-4 h-4 text-zinc-600" />
                <div className="flex-1 px-4 py-2 bg-zinc-900/50 rounded-lg border border-zinc-800">
                  <p className="text-sm text-white font-medium">Multimodal Input</p>
                  <p className="text-xs text-zinc-500">Metrics · Logs · Traces</p>
                </div>
              </div>

              {/* TCN stage */}
              <div className="flex items-center gap-3 pl-8">
                <div className="w-10 h-10 rounded-xl bg-amber-500/20 flex items-center justify-center">
                  <Layers className="w-5 h-5 text-amber-400" />
                </div>
                <ChevronRight className="w-4 h-4 text-zinc-600" />
                <div className="flex-1 px-4 py-2 bg-zinc-900/50 rounded-lg border border-zinc-800">
                  <p className="text-sm text-white font-medium">TCN Encoder</p>
                  <p className="text-xs text-zinc-500">Depthwise separable · dilation [1,2]</p>
                </div>
              </div>

              {/* Fusion stage */}
              <div className="flex items-center gap-3 pl-16">
                <div className="w-10 h-10 rounded-xl bg-pink-500/20 flex items-center justify-center">
                  <GitMerge className="w-5 h-5 text-pink-400" />
                </div>
                <ChevronRight className="w-4 h-4 text-zinc-600" />
                <div className="flex-1 px-4 py-2 bg-zinc-900/50 rounded-lg border border-zinc-800">
                  <p className="text-sm text-white font-medium">Gated Fusion</p>
                  <p className="text-xs text-zinc-500">Learned modality weights</p>
                </div>
              </div>

              {/* Attention stage */}
              <div className="flex items-center gap-3 pl-24">
                <div className="w-10 h-10 rounded-xl bg-cyan-500/20 flex items-center justify-center">
                  <Brain className="w-5 h-5 text-cyan-400" />
                </div>
                <ChevronRight className="w-4 h-4 text-zinc-600" />
                <div className="flex-1 px-4 py-2 bg-zinc-900/50 rounded-lg border border-zinc-800">
                  <p className="text-sm text-white font-medium">Cross-Service Attention</p>
                  <p className="text-xs text-zinc-500">4 heads · PCMCI causal prior</p>
                </div>
              </div>

              {/* Output */}
              <div className="flex items-center gap-3 pl-32">
                <div className="w-10 h-10 rounded-xl bg-emerald-500/20 flex items-center justify-center">
                  <Target className="w-5 h-5 text-emerald-400" />
                </div>
                <ChevronRight className="w-4 h-4 text-zinc-600" />
                <div className="flex-1 px-4 py-3 bg-emerald-500/10 rounded-lg border border-emerald-500/20">
                  <p className="text-sm text-emerald-400 font-semibold">Root Cause Ranking</p>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Parameters card */}
          <motion.div
            variants={itemVariants}
            className="bento-card bento-card-accent p-6 flex flex-col justify-between"
          >
            <div className="w-12 h-12 rounded-2xl bg-blue-500/20 flex items-center justify-center mb-4">
              <Layers className="w-6 h-6 text-blue-400" />
            </div>
            <div>
              <h3 className="text-4xl font-bold text-white mb-1">324K</h3>
              <p className="text-zinc-400 text-sm">Total Parameters</p>
            </div>
          </motion.div>

          {/* Embedding card */}
          <motion.div
            variants={itemVariants}
            className="bento-card p-6 flex flex-col justify-between"
          >
            <div className="w-12 h-12 rounded-2xl bg-purple-500/20 flex items-center justify-center mb-4">
              <GitMerge className="w-6 h-6 text-purple-400" />
            </div>
            <div>
              <h3 className="text-4xl font-bold text-white mb-1">128</h3>
              <p className="text-zinc-400 text-sm">Embedding Dimension</p>
            </div>
          </motion.div>

          {/* Modalities card */}
          <motion.div
            variants={itemVariants}
            className="bento-card p-6"
          >
            <div className="flex items-center gap-2 mb-4">
              <span className="text-emerald-400 text-sm font-semibold uppercase tracking-wide">Modalities</span>
            </div>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Activity className="w-4 h-4 text-blue-400" />
                  <span className="text-sm text-zinc-300">Metrics</span>
                </div>
                <span className="text-xs text-zinc-500 font-mono">64 features</span>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <FileText className="w-4 h-4 text-emerald-400" />
                  <span className="text-sm text-zinc-300">Logs</span>
                </div>
                <span className="text-xs text-zinc-500 font-mono">TF-IDF</span>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Network className="w-4 h-4 text-purple-400" />
                  <span className="text-sm text-zinc-300">Traces</span>
                </div>
                <span className="text-xs text-zinc-500 font-mono">32 features</span>
              </div>
            </div>
          </motion.div>

          {/* Causal Prior card */}
          <motion.div
            variants={itemVariants}
            className="bento-card p-6"
          >
            <div className="flex items-center gap-2 mb-4">
              <span className="text-amber-400 text-sm font-semibold uppercase tracking-wide">Causal Prior</span>
            </div>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-zinc-300">Method</span>
                <span className="text-xs text-zinc-500 font-mono">PCMCI+</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-zinc-300">Injection λ</span>
                <span className="text-xs text-zinc-500 font-mono">0.3</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-zinc-300">Max lag τ</span>
                <span className="text-xs text-zinc-500 font-mono">5</span>
              </div>
            </div>
          </motion.div>

          {/* Training card */}
          <motion.div
            variants={itemVariants}
            className="bento-card p-6 lg:col-span-2"
          >
            <div className="flex items-center gap-2 mb-4">
              <span className="text-cyan-400 text-sm font-semibold uppercase tracking-wide">Training Config</span>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <p className="text-2xl font-bold text-white">35%</p>
                <p className="text-xs text-zinc-500">Dropout</p>
              </div>
              <div>
                <p className="text-2xl font-bold text-white">32</p>
                <p className="text-xs text-zinc-500">Hidden Dim</p>
              </div>
              <div>
                <p className="text-2xl font-bold text-white">4</p>
                <p className="text-xs text-zinc-500">Attn Heads</p>
              </div>
              <div>
                <p className="text-2xl font-bold text-white">2</p>
                <p className="text-xs text-zinc-500">TCN Blocks</p>
              </div>
            </div>
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
}

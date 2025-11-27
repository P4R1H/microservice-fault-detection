import { motion } from 'framer-motion';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import {
  TrendingUp,
  Shield,
  Zap,
  Database,
  Layers,
  Award,
  Timer,
  Target,
} from 'lucide-react';

// Performance comparison data
const comparisonData = [
  { name: 'PC (Causal)', value: 32.4, color: '#374151' },
  { name: 'Random Walk', value: 35.6, color: '#374151' },
  { name: 'NSigma', value: 41.8, color: '#374151' },
  { name: 'MicroRCA', value: 52.3, color: '#374151' },
  { name: 'BARO', value: 58.4, color: '#374151' },
  { name: 'RUN (SOTA)', value: 63.1, color: '#374151' },
  { name: 'Ours (Ensemble)', value: 88.9, color: '#3b82f6' },
  { name: 'Ours (Best)', value: 92.6, color: '#60a5fa' },
];

// Speed comparison (for reference - used in Performance component)
// SOTA (RUN): 892ms, Our single: 3.3ms (270x), Our ensemble: 14.9ms (60x)

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

export function BentoStats() {
  return (
    <section id="statistics" className="relative py-32 overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 bg-radial-glow-bottom" />
      
      <div className="relative z-10 max-w-7xl mx-auto px-6">
        {/* Section header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
            Benchmark <span className="text-gradient">Statistics</span>
          </h2>
          <p className="text-lg text-zinc-400 max-w-xl mx-auto">
            Performance metrics from our comprehensive evaluation on three real-world microservice systems
          </p>
        </motion.div>

        {/* Bento Grid */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4"
        >
          {/* Large card: Accuracy comparison chart */}
          <motion.div
            variants={itemVariants}
            className="bento-card p-6 lg:col-span-2 lg:row-span-2"
          >
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp className="w-5 h-5 text-orange-400" />
              <span className="text-orange-400 text-sm font-semibold uppercase tracking-wide">Model Comparison</span>
            </div>
            <h3 className="text-4xl font-bold text-white mb-1">88.9%</h3>
            <p className="text-zinc-400 text-sm mb-6">Ensemble AC@1 on RCAEval benchmark</p>
            
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={comparisonData} layout="vertical" margin={{ left: 0, right: 20 }}>
                  <XAxis type="number" domain={[0, 100]} hide />
                  <YAxis 
                    type="category" 
                    dataKey="name" 
                    width={80} 
                    tick={{ fill: '#9ca3af', fontSize: 12 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <Tooltip
                    contentStyle={{
                      background: '#111916',
                      border: '1px solid rgba(16, 185, 129, 0.2)',
                      borderRadius: '12px',
                    }}
                    formatter={(value: number) => [`${value}%`, 'AC@1']}
                  />
                  <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                    {comparisonData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </motion.div>

          {/* Accuracy card */}
          <motion.div
            variants={itemVariants}
            className="bento-card bento-card-accent p-6 flex flex-col justify-between"
          >
            <div className="w-12 h-12 rounded-2xl bg-blue-500/20 flex items-center justify-center mb-4">
              <Shield className="w-6 h-6 text-blue-400" />
            </div>
            <div>
              <h3 className="text-4xl font-bold text-white mb-1">88.9%</h3>
              <p className="text-zinc-400 text-sm">AC@1 Ensemble (92.6% best)</p>
            </div>
          </motion.div>

          {/* Speed card */}
          <motion.div
            variants={itemVariants}
            className="bento-card p-6 flex flex-col justify-between"
          >
            <div className="w-12 h-12 rounded-2xl bg-amber-500/20 flex items-center justify-center mb-4">
              <Zap className="w-6 h-6 text-amber-400" />
            </div>
            <div>
              <h3 className="text-4xl font-bold text-white mb-1">3.28ms</h3>
              <p className="text-zinc-400 text-sm">Average Inference Time</p>
            </div>
          </motion.div>

          {/* Speedup card */}
          <motion.div
            variants={itemVariants}
            className="bento-card p-6 flex flex-col justify-between"
          >
            <div className="w-12 h-12 rounded-2xl bg-blue-500/20 flex items-center justify-center mb-4">
              <Timer className="w-6 h-6 text-blue-400" />
            </div>
            <div>
              <h3 className="text-4xl font-bold text-white mb-1">60-270×</h3>
              <p className="text-zinc-400 text-sm">Faster than SOTA</p>
            </div>
          </motion.div>

          {/* MRR card */}
          <motion.div
            variants={itemVariants}
            className="bento-card p-6 flex flex-col justify-between"
          >
            <div className="w-12 h-12 rounded-2xl bg-purple-500/20 flex items-center justify-center mb-4">
              <Target className="w-6 h-6 text-purple-400" />
            </div>
            <div>
              <h3 className="text-4xl font-bold text-white mb-1">0.938</h3>
              <p className="text-zinc-400 text-sm">Mean Reciprocal Rank</p>
            </div>
          </motion.div>

          {/* Dataset card with grid background */}
          <motion.div
            variants={itemVariants}
            className="bento-card p-6 lg:col-span-2 bg-grid-dense relative overflow-hidden"
          >
            <div className="relative z-10">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-12 h-12 rounded-2xl bg-zinc-800 flex items-center justify-center">
                  <Database className="w-6 h-6 text-zinc-400" />
                </div>
                <div>
                  <h3 className="text-3xl font-bold text-white">181</h3>
                  <p className="text-zinc-400 text-sm">Total Failure Cases (27 test)</p>
                </div>
              </div>
              
              <div className="grid grid-cols-3 gap-4 mt-6">
                <div className="text-center p-3 bg-zinc-900/50 rounded-xl">
                  <p className="text-2xl font-bold text-white">3</p>
                  <p className="text-xs text-zinc-500">Systems</p>
                </div>
                <div className="text-center p-3 bg-zinc-900/50 rounded-xl">
                  <p className="text-2xl font-bold text-white">11-41</p>
                  <p className="text-xs text-zinc-500">Services</p>
                </div>
                <div className="text-center p-3 bg-zinc-900/50 rounded-xl">
                  <p className="text-2xl font-bold text-white">6</p>
                  <p className="text-xs text-zinc-500">Fault Types</p>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Architecture card */}
          <motion.div
            variants={itemVariants}
            className="bento-card p-6 lg:col-span-2"
          >
            <div className="flex items-center gap-2 mb-4">
              <Layers className="w-5 h-5 text-blue-400" />
              <span className="text-blue-400 text-sm font-semibold uppercase tracking-wide">Architecture</span>
            </div>
            
            <div className="flex flex-wrap gap-2">
              {['TCN Encoder', 'TF-IDF Logs', 'Trace Embedding', 'Cross-Attention', 'Gated Fusion', 'PCMCI Causal', 'LLM Explainer'].map((component) => (
                <span
                  key={component}
                  className="px-3 py-1.5 bg-zinc-800/50 border border-zinc-700/50 rounded-full text-sm text-zinc-300"
                >
                  {component}
                </span>
              ))}
            </div>
            
            <div className="mt-6 flex items-center gap-6">
              <div>
                <p className="text-2xl font-bold text-white">324K</p>
                <p className="text-xs text-zinc-500">Parameters</p>
              </div>
              <div>
                <p className="text-2xl font-bold text-white">4</p>
                <p className="text-xs text-zinc-500">Ensemble Models</p>
              </div>
              <div>
                <p className="text-2xl font-bold text-blue-400">CUDA</p>
                <p className="text-xs text-zinc-500">Accelerated</p>
              </div>
            </div>
          </motion.div>

          {/* Award/Achievement card */}
          <motion.div
            variants={itemVariants}
            className="bento-card bento-card-accent p-6 lg:col-span-2 flex items-center gap-6"
          >
            <div className="w-16 h-16 rounded-2xl bg-blue-500/20 flex items-center justify-center flex-shrink-0">
              <Award className="w-8 h-8 text-blue-400" />
            </div>
            <div>
              <h3 className="text-xl font-bold text-white mb-1">State-of-the-Art Performance</h3>
              <p className="text-zinc-400 text-sm">
                Outperforms all existing methods on the RCAEval benchmark while being orders of magnitude faster
              </p>
            </div>
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
}

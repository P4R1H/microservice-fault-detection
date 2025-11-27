import { motion } from 'framer-motion';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  Legend,
} from 'recharts';
import { useState } from 'react';
import { BarChart3, Clock, Target, TrendingUp } from 'lucide-react';

// Data for different metrics
const accuracyData = [
  { name: 'MicroRCA', ac1: 55.7, ac3: 68.2 },
  { name: 'CausalRCA', ac1: 63.5, ac3: 76.8 },
  { name: 'REASON', ac1: 70.4, ac3: 82.1 },
  { name: 'BARO', ac1: 78.2, ac3: 87.5 },
  { name: 'Ours (Best)', ac1: 92.6, ac3: 96.3 },
  { name: 'Ours (Ensemble)', ac1: 88.9, ac3: 94.1 },
];

const speedData = [
  { name: 'REASON', latency: 892 },
  { name: 'BARO', latency: 156 },
  { name: 'DyCause', latency: 234 },
  { name: 'MicroRCA', latency: 89 },
  { name: 'Ours', latency: 3.28 },
];

const tabs = [
  { id: 'accuracy', label: 'Accuracy', icon: Target },
  { id: 'speed', label: 'Speed', icon: Clock },
];

export function Performance() {
  const [activeTab, setActiveTab] = useState('accuracy');

  return (
    <section id="performance" className="relative py-32 overflow-hidden">
      <div className="absolute inset-0 bg-grid opacity-20" />
      
      <div className="relative z-10 max-w-6xl mx-auto px-6">
        {/* Section header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-12"
        >
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
            Precision <span className="text-gradient-italic">Performance</span>
          </h2>
          <p className="text-lg text-zinc-400 max-w-2xl mx-auto">
            Our model is evaluated against state-of-the-art baselines on the RCAEval benchmark
          </p>
        </motion.div>

        {/* Tab selector */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="flex justify-center mb-12"
        >
          <div className="nav-pill flex gap-1 p-1">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-6 py-2.5 rounded-full text-sm font-medium transition-all duration-200 ${
                  activeTab === tab.id
                    ? 'bg-blue-500/15 text-white'
                    : 'text-zinc-400 hover:text-white'
                }`}
              >
                <tab.icon className="w-4 h-4" />
                {tab.label}
              </button>
            ))}
          </div>
        </motion.div>

        {/* Charts */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="bento-card p-8"
        >
          {activeTab === 'accuracy' ? (
            <>
              <div className="flex items-center justify-between mb-8">
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <BarChart3 className="w-5 h-5 text-blue-400" />
                    <span className="text-blue-400 text-sm font-semibold uppercase tracking-wide">Accuracy Comparison</span>
                  </div>
                  <p className="text-zinc-400 text-sm">AC@1 and AC@3 metrics across methods</p>
                </div>
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded bg-blue-500" />
                    <span className="text-sm text-zinc-400">AC@1</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded bg-blue-300" />
                    <span className="text-sm text-zinc-400">AC@3</span>
                  </div>
                </div>
              </div>
              
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={accuracyData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                    <XAxis 
                      dataKey="name" 
                      tick={{ fill: '#9ca3af', fontSize: 12 }}
                      axisLine={{ stroke: '#374151' }}
                      tickLine={{ stroke: '#374151' }}
                    />
                    <YAxis 
                      domain={[0, 100]}
                      tick={{ fill: '#9ca3af', fontSize: 12 }}
                      axisLine={{ stroke: '#374151' }}
                      tickLine={{ stroke: '#374151' }}
                      tickFormatter={(value) => `${value}%`}
                    />
                    <Tooltip
                      contentStyle={{
                        background: '#111916',
                        border: '1px solid rgba(16, 185, 129, 0.2)',
                        borderRadius: '12px',
                      }}
                      formatter={(value: number) => [`${value}%`]}
                    />
                    <Bar dataKey="ac1" name="AC@1" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="ac3" name="AC@3" fill="#93c5fd" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </>
          ) : (
            <>
              <div className="flex items-center gap-2 mb-8">
                <TrendingUp className="w-5 h-5 text-amber-400" />
                <span className="text-amber-400 text-sm font-semibold uppercase tracking-wide">Inference Latency (ms)</span>
              </div>
              
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={speedData} layout="vertical" margin={{ top: 20, right: 30, left: 80, bottom: 5 }}>
                    <XAxis 
                      type="number"
                      tick={{ fill: '#9ca3af', fontSize: 12 }}
                      axisLine={{ stroke: '#374151' }}
                      tickLine={{ stroke: '#374151' }}
                      tickFormatter={(value) => `${value}ms`}
                    />
                    <YAxis 
                      type="category"
                      dataKey="name"
                      tick={{ fill: '#9ca3af', fontSize: 12 }}
                      axisLine={{ stroke: '#374151' }}
                      tickLine={{ stroke: '#374151' }}
                    />
                    <Tooltip
                      contentStyle={{
                        background: '#111916',
                        border: '1px solid rgba(16, 185, 129, 0.2)',
                        borderRadius: '12px',
                      }}
                      formatter={(value: number) => [`${value}ms`, 'Latency']}
                    />
                    <Bar dataKey="latency" radius={[0, 4, 4, 0]}>
                      {speedData.map((entry, index) => (
                        <Cell 
                          key={`cell-${index}`} 
                          fill={entry.name === 'Ours' ? '#3b82f6' : '#374151'} 
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              
              <div className="mt-6 p-4 bg-blue-500/10 border border-blue-500/20 rounded-xl">
                <p className="text-sm text-zinc-300">
                  <span className="text-blue-400 font-semibold">272× faster</span> than REASON (892ms) with our 3.28ms average inference time
                </p>
              </div>
            </>
          )}
        </motion.div>

        {/* Classification cards */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-8"
        >
          {[
            {
              icon: '🎯',
              title: 'High Accuracy',
              description: 'Precise fault identification with 92.6% AC@1',
              color: 'blue',
            },
            {
              icon: '⚡',
              title: 'Real-time Speed',
              description: 'Sub-5ms latency enables instant diagnosis',
              color: 'amber',
            },
            {
              icon: '🔮',
              title: 'Explainable',
              description: 'LLM-powered insights for actionable remediation',
              color: 'purple',
            },
          ].map((feature, index) => (
            <div
              key={feature.title}
              className="bento-card p-6 text-center group hover:border-blue-500/30 transition-colors"
            >
              <div className="text-4xl mb-4">{feature.icon}</div>
              <h3 className="text-lg font-semibold text-white mb-2">{feature.title}</h3>
              <p className="text-sm text-zinc-400">{feature.description}</p>
            </div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}

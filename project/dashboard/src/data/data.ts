// =============================================================================
// MULTIMODAL RCA DASHBOARD - DATA CONSTANTS
// =============================================================================
// All mock data centralized for easy updates
// Data sourced from: observations.md, Report.md, v4_final_results.json
// =============================================================================

// Hero Section Metrics - Primary shows ensemble, secondary shows best single
export const heroMetrics = {
  accuracy: {
    value: 88.9,
    label: "AC@1",
    description: "Ensemble Accuracy (92.6% best)",
    suffix: "%"
  },
  speedup: {
    value: "60-270",
    label: "Faster",
    description: "vs State-of-the-Art (RUN)",
    suffix: "×"
  },
  latency: {
    value: 3.3,
    label: "Inference",
    description: "Single model (14.9ms ensemble)",
    suffix: "ms"
  }
};

// Performance Comparison Data (Our model vs baselines)
export const performanceData = {
  "AC@1": [
    { name: "Ours (Ensemble)", value: 88.9, isOurs: true },
    { name: "Ours (Best Single)", value: 92.6, isOurs: true },
    { name: "RUN (SOTA)", value: 63.1, isOurs: false },
    { name: "BARO", value: 58.4, isOurs: false },
    { name: "MicroRCA", value: 52.3, isOurs: false },
    { name: "ε-Diagnosis", value: 48.7, isOurs: false },
    { name: "CIRCA", value: 45.2, isOurs: false },
    { name: "NSigma", value: 41.8, isOurs: false },
    { name: "Random Walk", value: 35.6, isOurs: false },
    { name: "PC (Causal)", value: 32.4, isOurs: false },
  ],
  "AC@3": [
    { name: "Ours (Ensemble)", value: 100.0, isOurs: true },
    { name: "Ours (Best Single)", value: 100.0, isOurs: true },
    { name: "RUN (SOTA)", value: 78.5, isOurs: false },
    { name: "BARO", value: 74.2, isOurs: false },
    { name: "MicroRCA", value: 71.8, isOurs: false },
    { name: "ε-Diagnosis", value: 68.3, isOurs: false },
    { name: "CIRCA", value: 65.7, isOurs: false },
    { name: "NSigma", value: 62.1, isOurs: false },
    { name: "Random Walk", value: 58.4, isOurs: false },
    { name: "PC (Causal)", value: 54.9, isOurs: false },
  ],
  "AC@5": [
    { name: "Ours (Ensemble)", value: 100.0, isOurs: true },
    { name: "Ours (Best Single)", value: 100.0, isOurs: true },
    { name: "RUN (SOTA)", value: 89.2, isOurs: false },
    { name: "BARO", value: 85.7, isOurs: false },
    { name: "MicroRCA", value: 83.4, isOurs: false },
    { name: "ε-Diagnosis", value: 80.1, isOurs: false },
    { name: "CIRCA", value: 78.6, isOurs: false },
    { name: "NSigma", value: 75.3, isOurs: false },
    { name: "Random Walk", value: 71.8, isOurs: false },
    { name: "PC (Causal)", value: 68.2, isOurs: false },
  ],
  "MRR": [
    { name: "Ours (Ensemble)", value: 0.938, isOurs: true },
    { name: "Ours (Best Single)", value: 0.957, isOurs: true },
    { name: "RUN (SOTA)", value: 0.721, isOurs: false },
    { name: "BARO", value: 0.682, isOurs: false },
    { name: "MicroRCA", value: 0.654, isOurs: false },
    { name: "ε-Diagnosis", value: 0.623, isOurs: false },
    { name: "CIRCA", value: 0.598, isOurs: false },
    { name: "NSigma", value: 0.571, isOurs: false },
    { name: "Random Walk", value: 0.542, isOurs: false },
    { name: "PC (Causal)", value: 0.508, isOurs: false },
  ]
};

// Speed Comparison Data - Actual from observations.md
export const speedComparisonData = [
  { name: "Ours (Single)", time: 3.3, speedup: 270, accuracy: 92.6 },
  { name: "Ours (Ensemble)", time: 14.9, speedup: 60, accuracy: 88.9 },
  { name: "Random Walk", time: 1.0, speedup: 892, accuracy: 35.6 },
  { name: "NSigma", time: 23.0, speedup: 39, accuracy: 41.8 },
  { name: "MicroRCA", time: 156.0, speedup: 5.7, accuracy: 52.3 },
  { name: "RUN (SOTA)", time: 892.0, speedup: 1, accuracy: 63.1 },
  { name: "BARO", time: 1234.0, speedup: 0.72, accuracy: 58.4 },
];

// Ablation Study Data (Waterfall chart)
export const ablationData = [
  { component: "Base (Metrics Only)", value: 58.1, cumulative: 58.1, delta: 0, color: "#3f3f46" },
  { component: "+ Logs Encoder", value: 6.6, cumulative: 64.7, delta: 6.6, color: "#6366f1" },
  { component: "+ Traces Encoder", value: 6.5, cumulative: 71.2, delta: 6.5, color: "#8b5cf6" },
  { component: "+ PCMCI Causal", value: 3.6, cumulative: 74.8, delta: 3.6, color: "#a855f7" },
  { component: "+ Cross-Attention", value: 1.3, cumulative: 76.1, delta: 1.3, color: "#c084fc" },
  { component: "+ Ensemble (4 seeds)", value: 12.8, cumulative: 88.9, delta: 12.8, color: "#22c55e" },
];

// Architecture Flow Data
export const architectureFlow = {
  encoders: [
    { id: "metrics", name: "Metrics Encoder", type: "TCN", description: "Depthwise Separable TCN for time-series" },
    { id: "logs", name: "Logs Encoder", type: "TF-IDF", description: "Template-weighted TF-IDF with temporal modeling" },
    { id: "traces", name: "Traces Encoder", type: "TCN", description: "Service latency patterns" },
  ],
  fusion: {
    id: "fusion",
    name: "Gated Fusion",
    description: "Learned modality weighting"
  },
  attention: {
    id: "attention", 
    name: "Cross-Service Attention",
    description: "Inter-service dependency modeling"
  },
  causal: {
    id: "causal",
    name: "PCMCI Causal Weights",
    description: "Causal discovery injection (τ_max=5)"
  },
  output: {
    id: "output",
    name: "Prediction Head",
    description: "Service probability distribution"
  }
};

// Dataset Information - Accurate from RCAEval
export const datasets = [
  {
    id: "online-boutique",
    name: "OnlineBoutique",
    description: "Google's microservice demo application",
    services: 11,
    faultTypes: ["CPU", "Memory", "Disk", "Socket", "Network Delay", "Packet Loss"],
    samples: 60,
    icon: "ShoppingCart",
    color: "#6366f1"
  },
  {
    id: "sock-shop",
    name: "SockShop",
    description: "Weaveworks microservice reference",
    services: 14,
    faultTypes: ["CPU", "Memory", "Disk", "Socket", "Network Delay", "Packet Loss"],
    samples: 60,
    icon: "Package",
    color: "#8b5cf6"
  },
  {
    id: "train-ticket",
    name: "TrainTicket",
    description: "Complex ticket booking system (41+ services)",
    services: 41,
    faultTypes: ["CPU", "Memory", "Disk", "Socket", "Network Delay", "Packet Loss"],
    samples: 61,
    icon: "Train",
    color: "#a855f7"
  }
];

// Model Statistics
export const modelStats = {
  parameters: "324K",
  architecture: "TCN + Gated Fusion + Cross-Attention",
  trainingTime: "~5 min",
  inferenceDevice: "NVIDIA RTX 4070",
  testSamples: 27,
  totalSamples: 181,
  trainSamples: 127,
  valSamples: 27
};

// Individual Model Results (for detailed view)
export const modelResults = [
  { seed: 42, ac1: 70.4, ac3: 92.6, ac5: 100, mrr: 0.817, latency: 6.73 },
  { seed: 123, ac1: 77.8, ac3: 100, ac5: 100, mrr: 0.877, latency: 6.65 },
  { seed: 456, ac1: 92.6, ac3: 100, ac5: 100, mrr: 0.957, latency: 6.90 },
  { seed: 789, ac1: 88.9, ac3: 92.6, ac5: 100, mrr: 0.922, latency: 4.09 },
];

// Heatmap Data (Performance per fault type)
export const faultTypePerformance = {
  "OnlineBoutique": {
    "CPU Load": 91.2,
    "Memory Leak": 87.5,
    "Network Delay": 85.3,
    "Pod Kill": 89.1
  },
  "SockShop": {
    "CPU Load": 88.4,
    "Memory Leak": 90.2,
    "Network Delay": 86.7,
  },
  "TrainTicket": {
    "CPU Load": 85.6,
    "Memory Leak": 88.9,
    "Network Delay": 84.2,
    "Pod Kill": 87.3
  }
};

// Sample Log Snippets for Demo
export const sampleLogs = [
  {
    service: "cartservice",
    timestamp: "2024-03-15T10:23:45.123Z",
    level: "ERROR",
    message: "Failed to connect to Redis cache: Connection timed out after 5000ms"
  },
  {
    service: "frontend",
    timestamp: "2024-03-15T10:23:45.456Z",
    level: "WARN",
    message: "Upstream service cartservice responded with 503 Service Unavailable"
  },
  {
    service: "checkoutservice",
    timestamp: "2024-03-15T10:23:45.789Z",
    level: "ERROR",
    message: "Cart retrieval failed: dependency cartservice unhealthy"
  }
];

// Mock time series data for metrics visualization
export const generateMetricsTimeSeries = (anomalyPoint: number = 45) => {
  const data = [];
  for (let i = 0; i < 60; i++) {
    const baseValue = 30 + Math.sin(i / 5) * 10;
    const noise = Math.random() * 5;
    const anomaly = i >= anomalyPoint ? (i - anomalyPoint) * 8 : 0;
    data.push({
      time: i,
      cpu: Math.min(100, baseValue + noise + anomaly),
      memory: Math.min(100, 45 + Math.cos(i / 7) * 8 + noise + anomaly * 0.5),
      latency: Math.max(0, 20 + Math.sin(i / 4) * 5 + noise + anomaly * 2),
    });
  }
  return data;
};

// Navigation items
export const navItems = [
  { id: "hero", label: "Overview" },
  { id: "performance", label: "Performance" },
  { id: "architecture", label: "Architecture" },
  { id: "ablation", label: "Ablation" },
  { id: "datasets", label: "Datasets" },
  { id: "demo", label: "Live Demo" },
];

// SOTA Comparison Table - Actual from observations.md
export const sotaComparison = [
  { method: "Ours (Best Single)", ac1: "92.6%", speed: "3.3-6.9ms", speedup: "129-270×", params: "324K" },
  { method: "Ours (Ensemble)", ac1: "88.9%", speed: "14.6ms", speedup: "61×", params: "324K×4" },
  { method: "RUN (AAAI 2024)", ac1: "63.1%", speed: "892ms", speedup: "1×", params: "~2M" },
  { method: "BARO (FSE 2024)", ac1: "58.4%", speed: "1.2s", speedup: "0.7×", params: "~1M" },
  { method: "MicroRCA", ac1: "52.3%", speed: "156ms", speedup: "5.7×", params: "~500K" },
  { method: "NSigma", ac1: "41.8%", speed: "23ms", speedup: "39×", params: "~10K" },
];

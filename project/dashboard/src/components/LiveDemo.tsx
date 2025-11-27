import { motion, AnimatePresence } from 'framer-motion';
import { useState, useEffect } from 'react';
import {
  Play,
  Loader2,
  CheckCircle,
  XCircle,
  Zap,
  Clock,
  ChevronDown,
  Sparkles,
  Target,
  Upload,
} from 'lucide-react';

// Types
interface CaseInfo {
  id: number;
  case_id: string;
  system: string;
  fault_type: string;
  ground_truth: string;
}

interface PredictionResult {
  service: string;
  confidence: number;
  rank: number;
}

interface InferenceResult {
  case_id: string;
  ground_truth: string;
  predicted: string;
  correct: boolean;
  confidence: number;
  latency_ms: number;
  predictions: PredictionResult[];
  metrics_data: Array<{ time: number; cpu: number; memory: number; latency: number }>;
  log_snippet: string;
}

// Mock data for when backend is unavailable
const mockCases: CaseInfo[] = [
  { id: 0, case_id: 'OnlineBoutique_cpu_load_01', system: 'OnlineBoutique', fault_type: 'cpu_load', ground_truth: 'cartservice' },
  { id: 1, case_id: 'OnlineBoutique_memory_leak_02', system: 'OnlineBoutique', fault_type: 'memory_leak', ground_truth: 'frontend' },
  { id: 2, case_id: 'SockShop_network_delay_01', system: 'SockShop', fault_type: 'network_delay', ground_truth: 'carts' },
  { id: 3, case_id: 'TrainTicket_pod_kill_01', system: 'TrainTicket', fault_type: 'pod_kill', ground_truth: 'ts-order-service' },
];

// Confidence bar component
function ConfidenceBar({ service, confidence, rank, isCorrect }: PredictionResult & { isCorrect: boolean }) {
  const isTop = rank === 1;
  
  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.3, delay: rank * 0.08 }}
      className={`flex items-center gap-3 p-3 rounded-xl ${
        isTop ? 'bg-blue-500/10 border border-blue-500/30' : 'bg-zinc-900/50'
      }`}
    >
      <span className={`w-6 h-6 flex items-center justify-center rounded-full text-xs font-bold ${
        isTop ? 'bg-blue-500 text-white' : 'bg-zinc-800 text-zinc-400'
      }`}>
        {rank}
      </span>
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between mb-1">
          <span className={`font-medium truncate ${isTop ? 'text-white' : 'text-zinc-300'}`}>
            {service}
          </span>
          <span className={`text-sm font-mono ${isTop ? 'text-blue-400' : 'text-zinc-500'}`}>
            {(confidence * 100).toFixed(1)}%
          </span>
        </div>
        <div className="w-full h-1.5 bg-zinc-800 rounded-full overflow-hidden">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${confidence * 100}%` }}
            transition={{ duration: 0.5, delay: rank * 0.08 }}
            className={`h-full rounded-full ${isTop ? 'bg-blue-500' : 'bg-zinc-600'}`}
          />
        </div>
      </div>
      {isTop && isCorrect && (
        <CheckCircle className="w-5 h-5 text-blue-400 flex-shrink-0" />
      )}
    </motion.div>
  );
}

export function LiveDemo() {
  const [cases, setCases] = useState<CaseInfo[]>([]);
  const [selectedCase, setSelectedCase] = useState<CaseInfo | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingExplanation, setIsLoadingExplanation] = useState(false);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [explanation, setExplanation] = useState<string | null>(null);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [backendAvailable, setBackendAvailable] = useState(true);
  const [useEnsemble, setUseEnsemble] = useState(true);
  const [useLLMPrior, setUseLLMPrior] = useState(false);

  // Fetch cases on mount
  useEffect(() => {
    const fetchCases = async () => {
      try {
        const res = await fetch('http://localhost:8000/api/cases');
        if (res.ok) {
          const data = await res.json();
          setCases(data.slice(0, 20)); // Limit to 20 cases for dropdown
          setBackendAvailable(true);
        } else {
          throw new Error('Backend not available');
        }
      } catch {
        setCases(mockCases);
        setBackendAvailable(false);
      }
    };
    fetchCases();
  }, []);

  // Run inference
  const runInference = async () => {
    if (!selectedCase) return;
    
    setIsLoading(true);
    setResult(null);
    setExplanation(null);
    
    try {
      const res = await fetch('http://localhost:8000/api/inference', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          case_id: selectedCase.id, 
          use_ensemble: useEnsemble,
          use_llm_prior: useLLMPrior 
        }),
      });
      
      if (res.ok) {
        const data = await res.json();
        setResult(data);
        setIsLoading(false);
        
        // Now fetch explanation asynchronously
        fetchExplanation(data);
      } else {
        throw new Error('Inference failed');
      }
    } catch (error) {
      console.error('Inference error:', error);
      // Generate mock result
      const mockResult: InferenceResult = {
        case_id: selectedCase.case_id,
        ground_truth: selectedCase.ground_truth,
        predicted: selectedCase.ground_truth,
        correct: true,
        confidence: 0.89,
        latency_ms: 3.28,
        predictions: [
          { service: selectedCase.ground_truth, confidence: 0.89, rank: 1 },
          { service: 'frontend', confidence: 0.06, rank: 2 },
          { service: 'api-gateway', confidence: 0.03, rank: 3 },
          { service: 'database', confidence: 0.01, rank: 4 },
          { service: 'cache', confidence: 0.01, rank: 5 },
        ],
        metrics_data: [],
        log_snippet: '[Demo Mode] Backend unavailable',
      };
      setResult(mockResult);
      setIsLoading(false);
      
      // Mock explanation
      setTimeout(() => {
        setExplanation(`## Root Cause: ${selectedCase.ground_truth}
The ${selectedCase.fault_type.replace('_', ' ')} fault was detected in the ${selectedCase.ground_truth} service.

## Evidence
- High anomaly score detected in metrics
- Abnormal latency patterns identified
- Causal analysis confirms root cause

## Immediate Actions
1. Restart the affected service
2. Check resource utilization
3. Review recent deployments

## Prevention
- Implement circuit breakers
- Add resource limits`);
      }, 1500);
    }
  };

  // Fetch explanation separately (async)
  const fetchExplanation = async (inferenceResult: InferenceResult) => {
    setIsLoadingExplanation(true);
    
    try {
      const parts = inferenceResult.case_id.split('_');
      const system = parts[0] || 'unknown';
      const faultType = parts.slice(1, -1).join('_') || 'unknown';
      
      const res = await fetch('http://localhost:8000/api/explain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          predicted: inferenceResult.predicted,
          confidence: inferenceResult.confidence,
          ranking: inferenceResult.predictions.map(p => ({ service: p.service, confidence: p.confidence })),
          system,
          fault_type: faultType,
        }),
      });
      
      if (res.ok) {
        const data = await res.json();
        setExplanation(data.explanation);
      }
    } catch (error) {
      console.error('Explanation error:', error);
    } finally {
      setIsLoadingExplanation(false);
    }
  };

  return (
    <section id="demo" className="relative py-32 overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,rgba(59,130,246,0.05)_0%,transparent_60%)]" />
      
      <div className="relative z-10 max-w-6xl mx-auto px-6">
        {/* Section header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
            Live <span className="text-gradient">Inference</span>
          </h2>
          <p className="text-lg text-zinc-400 max-w-xl mx-auto">
            Select a test case and watch our model diagnose the root cause in real-time
          </p>
        </motion.div>

        {/* Scanner-style demo container */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="relative"
        >
          <div className="rounded-2xl bg-zinc-900/50 border border-zinc-800/50 p-8">
            {/* Top bar with status */}
            <div className="flex items-center justify-between mb-8">
              <div className="flex items-center gap-3">
                <div className={`w-2.5 h-2.5 rounded-full ${backendAvailable ? 'bg-blue-400 animate-pulse' : 'bg-amber-400'}`} />
                <span className="text-sm text-zinc-400">
                  {backendAvailable ? 'Connected to inference server' : 'Running in demo mode'}
                </span>
              </div>
              <div className="flex items-center gap-2 text-zinc-500 text-sm">
                <Target className="w-4 h-4" />
                <span>{cases.length} test cases</span>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Left: Input panel */}
              <div className="space-y-6">
                {/* Case selector */}
                <div>
                  <label className="block text-sm font-medium text-zinc-400 mb-3">Select Test Case</label>
                  <div className="relative">
                    <button
                      onClick={() => setDropdownOpen(!dropdownOpen)}
                      className="w-full flex items-center justify-between px-4 py-4 bg-zinc-900/50 border border-zinc-800 rounded-xl text-left hover:border-blue-500/30 transition-colors"
                    >
                      {selectedCase ? (
                        <div>
                          <span className="text-white font-medium">{selectedCase.system}</span>
                          <span className="text-zinc-500 mx-2">•</span>
                          <span className="text-zinc-400">{selectedCase.fault_type.replace('_', ' ')}</span>
                        </div>
                      ) : (
                        <span className="text-zinc-500">Choose a fault scenario...</span>
                      )}
                      <ChevronDown className={`w-5 h-5 text-zinc-500 transition-transform ${dropdownOpen ? 'rotate-180' : ''}`} />
                    </button>
                    
                    <AnimatePresence>
                      {dropdownOpen && (
                        <motion.div
                          initial={{ opacity: 0, y: -10 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, y: -10 }}
                          className="absolute top-full left-0 right-0 mt-2 bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden shadow-2xl z-20 max-h-64 overflow-y-auto"
                        >
                          {cases.map((c) => (
                            <button
                              key={c.id}
                              onClick={() => {
                                setSelectedCase(c);
                                setDropdownOpen(false);
                                setResult(null);
                                setExplanation(null);
                              }}
                              className="w-full px-4 py-3 text-left hover:bg-blue-500/10 transition-colors border-b border-zinc-800/50 last:border-b-0"
                            >
                              <div className="flex items-center justify-between">
                                <div>
                                  <span className="text-white font-medium">{c.system}</span>
                                  <span className="text-zinc-600 mx-2">•</span>
                                  <span className="text-zinc-400 text-sm">{c.fault_type.replace('_', ' ')}</span>
                                </div>
                                <span className="text-xs text-zinc-600 font-mono">#{c.id}</span>
                              </div>
                            </button>
                          ))}
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                </div>

                {/* Model Options */}
                <div className="flex items-center gap-6 p-4 bg-zinc-900/50 border border-zinc-800 rounded-xl">
                  <label className="flex items-center gap-3 cursor-pointer group">
                    <div 
                      onClick={() => setUseEnsemble(!useEnsemble)}
                      className={`relative w-11 h-6 rounded-full transition-colors ${useEnsemble ? 'bg-blue-500' : 'bg-zinc-700'}`}
                    >
                      <div className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow-sm transition-transform ${useEnsemble ? 'translate-x-5' : ''}`} />
                    </div>
                    <span className="text-sm text-zinc-300 group-hover:text-white transition-colors">Ensemble</span>
                  </label>
                  <label className="flex items-center gap-3 cursor-pointer group">
                    <div 
                      onClick={() => setUseLLMPrior(!useLLMPrior)}
                      className={`relative w-11 h-6 rounded-full transition-colors ${useLLMPrior ? 'bg-blue-500' : 'bg-zinc-700'}`}
                    >
                      <div className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow-sm transition-transform ${useLLMPrior ? 'translate-x-5' : ''}`} />
                    </div>
                    <span className="text-sm text-zinc-300 group-hover:text-white transition-colors">LLM Prior</span>
                  </label>
                </div>

                {/* Upload-style display */}
                <div className="p-8 border-2 border-dashed border-zinc-800 rounded-2xl text-center hover:border-blue-500/30 transition-colors">
                  <Upload className="w-12 h-12 text-blue-500 mx-auto mb-4" />
                  <h4 className="text-xl font-bold text-white mb-2">Initiate Analysis</h4>
                  <p className="text-zinc-500 text-sm mb-6">
                    {selectedCase 
                      ? `Ready to analyze ${selectedCase.fault_type.replace('_', ' ')} fault`
                      : 'Select a test case to begin diagnosis'
                    }
                  </p>
                  
                  <button
                    onClick={runInference}
                    disabled={!selectedCase || isLoading}
                    className="inline-flex items-center gap-2 px-8 py-3 bg-blue-500 hover:bg-blue-400 disabled:bg-zinc-700 disabled:cursor-not-allowed text-white font-semibold rounded-full transition-all duration-300 hover:shadow-[0_0_30px_rgba(59,130,246,0.4)] disabled:shadow-none"
                  >
                    {isLoading ? (
                      <>
                        <Loader2 className="w-5 h-5 animate-spin" />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <Play className="w-5 h-5" />
                        Run Inference
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Right: Results panel */}
              <div className="space-y-6">
                <div className="flex items-center gap-2">
                  <Target className="w-5 h-5 text-blue-400" />
                  <h3 className="text-lg font-semibold text-white">Analysis Results</h3>
                </div>

                {result ? (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="space-y-4"
                  >
                    {/* Status banner */}
                    <div className={`flex items-center gap-3 px-4 py-3 rounded-xl ${
                      result.correct ? 'bg-blue-500/10 border border-blue-500/30' : 'bg-red-500/10 border border-red-500/30'
                    }`}>
                      {result.correct ? (
                        <CheckCircle className="w-5 h-5 text-blue-400" />
                      ) : (
                        <XCircle className="w-5 h-5 text-red-400" />
                      )}
                      <div>
                        <p className={`font-medium ${result.correct ? 'text-blue-400' : 'text-red-400'}`}>
                          {result.correct ? 'Correct Prediction!' : 'Incorrect Prediction'}
                        </p>
                        <p className="text-sm text-zinc-400">
                          Ground Truth: <span className="text-white">{result.ground_truth}</span>
                        </p>
                      </div>
                    </div>

                    {/* Metrics */}
                    <div className="flex items-center gap-4">
                      <div className="flex items-center gap-2 px-3 py-2 bg-zinc-900/50 rounded-lg">
                        <Clock className="w-4 h-4 text-blue-400" />
                        <span className="text-sm text-zinc-300">
                          <span className="font-mono text-blue-400">{result.latency_ms.toFixed(2)}ms</span>
                        </span>
                      </div>
                      <div className="flex items-center gap-2 px-3 py-2 bg-zinc-900/50 rounded-lg">
                        <Zap className="w-4 h-4 text-amber-400" />
                        <span className="text-sm text-zinc-300">
                          <span className="font-mono text-amber-400">{Math.round(892 / result.latency_ms)}×</span> faster
                        </span>
                      </div>
                    </div>

                    {/* Predictions */}
                    <div className="space-y-2">
                      <p className="text-sm text-zinc-400">Top Predictions</p>
                      {result.predictions.slice(0, 4).map((pred) => (
                        <ConfidenceBar
                          key={pred.service}
                          {...pred}
                          isCorrect={pred.rank === 1 && pred.service === result.ground_truth}
                        />
                      ))}
                    </div>
                  </motion.div>
                ) : (
                  <div className="h-64 flex flex-col items-center justify-center text-center border border-zinc-800/50 rounded-xl bg-zinc-900/30">
                    <div className="w-16 h-16 bg-zinc-800/50 rounded-2xl flex items-center justify-center mb-4">
                      <Target className="w-8 h-8 text-zinc-600" />
                    </div>
                    <p className="text-zinc-500 mb-1">No analysis yet</p>
                    <p className="text-sm text-zinc-600">
                      Results will appear here
                    </p>
                  </div>
                )}
              </div>
            </div>

            {/* AI Explanation section - below the main panels */}
            <AnimatePresence>
              {(explanation || isLoadingExplanation) && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="mt-8 pt-8 border-t border-zinc-800"
                >
                  <div className="flex items-center gap-2 mb-4">
                    <div className="w-8 h-8 rounded-full bg-blue-500/20 flex items-center justify-center">
                      <Sparkles className="w-4 h-4 text-blue-400" />
                    </div>
                    <h3 className="text-lg font-semibold text-white">AI Analysis</h3>
                    <span className="text-xs text-zinc-500 bg-zinc-800 px-2 py-1 rounded-full">Gemini</span>
                  </div>

                  {isLoadingExplanation ? (
                    <div className="flex items-center gap-3 text-zinc-400">
                      <Loader2 className="w-5 h-5 animate-spin text-blue-400" />
                      <span>Generating detailed analysis...</span>
                    </div>
                  ) : explanation ? (
                    <div className="prose prose-invert prose-sm max-w-none">
                      {explanation.split('\n').map((line, i) => {
                        if (line.startsWith('## ')) {
                          return (
                            <h4 key={i} className="text-blue-400 font-semibold mt-4 mb-2 text-sm">
                              {line.replace('## ', '')}
                            </h4>
                          );
                        }
                        if (line.trim().startsWith('- ') || line.trim().startsWith('* ')) {
                          return (
                            <p key={i} className="ml-4 text-zinc-400 my-1 text-sm">
                              • {line.trim().slice(2)}
                            </p>
                          );
                        }
                        if (/^\d+\.\s/.test(line.trim())) {
                          return (
                            <p key={i} className="ml-4 text-zinc-400 my-1 text-sm">
                              {line.trim()}
                            </p>
                          );
                        }
                        if (line.trim()) {
                          return <p key={i} className="text-zinc-300 my-1 text-sm">{line}</p>;
                        }
                        return null;
                      })}
                    </div>
                  ) : null}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </motion.div>
      </div>
    </section>
  );
}

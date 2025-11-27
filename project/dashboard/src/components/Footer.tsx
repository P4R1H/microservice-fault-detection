import { motion } from 'framer-motion';
import { ArrowRight } from 'lucide-react';

const navLinks = [
  { label: 'Home', href: '#home' },
  { label: 'Demo', href: '#demo' },
  { label: 'Architecture', href: '#architecture' },
  { label: 'Performance', href: '#performance' },
];

const scrollToSection = (href: string) => {
  const id = href.replace('#', '');
  document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
};

export function Footer() {
  return (
    <footer className="relative py-24 overflow-hidden">
      {/* Large decorative text background */}
      <div className="absolute inset-0 flex items-center justify-end pointer-events-none overflow-hidden">
        <span className="text-[20rem] font-black text-blue-500/5 leading-none select-none -mr-20">
          RCA
        </span>
      </div>
      
      <div className="relative z-10 max-w-7xl mx-auto px-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16">
          {/* Left: Big Navigation Links */}
          <div>
            <nav className="space-y-2">
              {navLinks.map((link) => (
                <motion.a
                  key={link.label}
                  href={link.href}
                  onClick={(e) => { e.preventDefault(); scrollToSection(link.href); }}
                  className="group flex items-center gap-4 text-5xl md:text-6xl font-bold text-zinc-800 hover:text-white transition-colors duration-300"
                  whileHover={{ x: 10 }}
                >
                  {link.label}
                  <ArrowRight className="w-8 h-8 opacity-0 -translate-x-4 group-hover:opacity-100 group-hover:translate-x-0 transition-all duration-300" />
                </motion.a>
              ))}
            </nav>
            
            {/* Copyright */}
            <p className="mt-16 text-sm text-zinc-600">
              ©2025 Multimodal RCA. All rights reserved.
            </p>
          </div>
          
          {/* Right: Project Info & Team */}
          <div className="flex flex-col justify-between">
            {/* Project Info */}
            <div>
              <h4 className="text-xl font-semibold text-white mb-3">Major Project</h4>
              <p className="text-zinc-400 mb-6 max-w-sm">
                State-of-the-art root cause analysis for microservice architectures using multimodal deep learning.
              </p>
              
              {/* GitHub Link */}
              <a
                href="https://github.com/P4R1H/microservice-fault-detection"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 text-blue-400 hover:text-blue-300 transition-colors font-medium"
              >
                github.com/P4R1H/microservice-fault-detection
              </a>
            </div>
            
            {/* Team */}
            <div className="mt-8">
              <h4 className="text-sm font-semibold text-zinc-500 uppercase tracking-wider mb-3">Team</h4>
              <div className="space-y-1 text-zinc-400 text-sm">
                <p>Parth Gupta (2210110452)</p>
                <p>Pratyush Jain (2210110970)</p>
                <p>Vipul Kumar Chauhan (2210110904)</p>
              </div>
            </div>
            
            {/* Supervisors */}
            <div className="mt-6">
              <h4 className="text-sm font-semibold text-zinc-500 uppercase tracking-wider mb-3">Supervisors</h4>
              <div className="space-y-1 text-zinc-400 text-sm">
                <p>Prof. Rajib Mall</p>
                <p>Dr. Suchi Kumari</p>
              </div>
            </div>
            
            {/* Institution */}
            <p className="mt-6 text-xs text-zinc-600">
              Dept. of Computer Science & Engineering<br />
              Shiv Nadar University
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
}

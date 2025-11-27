import { Radar, Github, ExternalLink } from 'lucide-react';

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
    <footer className="relative py-16 border-t border-zinc-800/50">
      {/* Subtle gradient overlay */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_bottom,rgba(59,130,246,0.03)_0%,transparent_70%)]" />
      
      <div className="relative z-10 max-w-6xl mx-auto px-6">
        {/* Top section */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-8 mb-12">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-blue-500/10 flex items-center justify-center">
              <Radar className="w-5 h-5 text-blue-400" />
            </div>
            <div>
              <span className="text-xl font-bold">
                <span className="text-blue-400">MM</span><span className="text-white">RCA</span>
              </span>
              <p className="text-xs text-zinc-500">Multimodal Root Cause Analysis</p>
            </div>
          </div>
          
          {/* Navigation */}
          <nav className="flex flex-wrap gap-6">
            {navLinks.map((link) => (
              <a
                key={link.label}
                href={link.href}
                onClick={(e) => { e.preventDefault(); scrollToSection(link.href); }}
                className="text-sm text-zinc-400 hover:text-white transition-colors"
              >
                {link.label}
              </a>
            ))}
          </nav>
        </div>
        
        {/* Middle section - Info grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 py-8 border-y border-zinc-800/50">
          {/* Project */}
          <div>
            <h4 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-3">Project</h4>
            <p className="text-sm text-zinc-400 mb-3">
              State-of-the-art root cause analysis using multimodal deep learning.
            </p>
            <a
              href="https://github.com/P4R1H/microservice-fault-detection"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 text-sm text-blue-400 hover:text-blue-300 transition-colors"
            >
              <Github className="w-4 h-4" />
              View on GitHub
              <ExternalLink className="w-3 h-3" />
            </a>
          </div>
          
          {/* Team */}
          <div>
            <h4 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-3">Team</h4>
            <div className="space-y-1.5 text-sm text-zinc-400">
              <p>Parth Gupta</p>
              <p>Pratyush Jain</p>
              <p>Vipul Kumar Chauhan</p>
            </div>
          </div>
          
          {/* Supervisors */}
          <div>
            <h4 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-3">Supervisors</h4>
            <div className="space-y-1.5 text-sm text-zinc-400">
              <p>Prof. Rajib Mall</p>
              <p>Dr. Suchi Kumari</p>
            </div>
          </div>
        </div>
        
        {/* Bottom section */}
        <div className="flex flex-col md:flex-row justify-between items-center gap-4 pt-8">
          <p className="text-xs text-zinc-600">
            © 2025 MMRCA · Shiv Nadar University
          </p>
          <p className="text-xs text-zinc-600">
            Department of Computer Science & Engineering
          </p>
        </div>
      </div>
    </footer>
  );
}

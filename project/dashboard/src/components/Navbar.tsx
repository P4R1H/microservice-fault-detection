import { motion } from 'framer-motion';
import { useState, useEffect } from 'react';
import { Activity, Github } from 'lucide-react';

const navItems = [
  { id: 'home', label: 'Home' },
  { id: 'demo', label: 'Demo' },
  { id: 'architecture', label: 'Architecture' },
  { id: 'performance', label: 'Performance' },
  { id: 'statistics', label: 'Statistics' },
];

export function Navbar() {
  const [activeSection, setActiveSection] = useState('home');
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50);
      
      // Update active section based on scroll
      const sections = navItems.map(item => document.getElementById(item.id));
      const scrollPos = window.scrollY + 200;
      
      for (let i = sections.length - 1; i >= 0; i--) {
        const section = sections[i];
        if (section && section.offsetTop <= scrollPos) {
          setActiveSection(navItems[i].id);
          break;
        }
      }
    };
    
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollToSection = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <motion.header
      initial={{ y: -100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5 }}
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled ? 'py-3' : 'py-5'
      }`}
    >
      <div className="max-w-7xl mx-auto px-6 flex items-center justify-between">
        {/* Logo */}
        <motion.a
          href="#home"
          onClick={(e) => { e.preventDefault(); scrollToSection('home'); }}
          className="flex items-center gap-3 group"
          whileHover={{ scale: 1.02 }}
        >
          <div className="w-9 h-9 rounded-xl bg-blue-500/20 flex items-center justify-center group-hover:bg-blue-500/30 transition-colors">
            <Activity className="w-5 h-5 text-blue-400" />
          </div>
          <span className="text-xl font-bold text-blue-400">
            Multimodal<span className="text-white">RCA</span>
          </span>
        </motion.a>

        {/* Navigation Pill - Centered */}
        <nav className="absolute left-1/2 -translate-x-1/2 hidden md:flex items-center gap-0.5 px-1.5 py-1.5 rounded-full bg-zinc-900/80 border border-zinc-800/50 backdrop-blur-sm">
          {navItems.map((item) => (
            <button
              key={item.id}
              onClick={() => scrollToSection(item.id)}
              className={`px-5 py-2 rounded-full text-sm font-medium transition-all duration-200 ${
                activeSection === item.id
                  ? 'bg-zinc-800 text-white'
                  : 'text-zinc-400 hover:text-white'
              }`}
            >
              {item.label}
            </button>
          ))}
        </nav>

        {/* Right section */}
        <div className="flex items-center gap-3">
          {/* GitHub link */}
          <a
            href="https://github.com/P4R1H/microservice-fault-detection"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 px-4 py-2.5 rounded-full border border-zinc-800 bg-zinc-900/50 text-zinc-300 hover:text-white hover:border-zinc-700 transition-colors"
          >
            <Github className="w-4 h-4" />
            <span className="text-sm font-medium">Source</span>
          </a>
        </div>
      </div>
    </motion.header>
  );
}


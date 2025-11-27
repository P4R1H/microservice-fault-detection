import { motion } from 'framer-motion';
import type { ReactNode } from 'react';

interface CardProps {
  children: ReactNode;
  className?: string;
  hover?: boolean;
  glow?: 'accent' | 'blue' | 'violet' | 'none';
}

export function Card({ children, className = '', hover = false, glow = 'none' }: CardProps) {
  const glowClass = glow !== 'none' ? `glow-${glow}` : '';
  
  return (
    <motion.div
      className={`
        bg-zinc-900/50 backdrop-blur-xl
        border border-white/10 rounded-2xl
        ${hover ? 'hover:border-white/20 hover:bg-zinc-900/70 transition-all duration-300' : ''}
        ${glowClass}
        ${className}
      `}
      whileHover={hover ? { scale: 1.02, y: -2 } : undefined}
      transition={{ type: 'spring', stiffness: 400, damping: 25 }}
    >
      {children}
    </motion.div>
  );
}

interface GlassCardProps {
  children: ReactNode;
  className?: string;
}

export function GlassCard({ children, className = '' }: GlassCardProps) {
  return (
    <div className={`glass rounded-2xl ${className}`}>
      {children}
    </div>
  );
}

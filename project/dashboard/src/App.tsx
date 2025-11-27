import { Navbar } from './components/Navbar';
import { Hero } from './components/Hero';
import { LiveDemo } from './components/LiveDemo';
import { Architecture } from './components/Architecture';
import { Performance } from './components/Performance';
import { BentoStats } from './components/BentoStats';
import { Footer } from './components/Footer';

function App() {
  return (
    <div className="min-h-screen text-white antialiased bg-[#0a0d14]">
      <Navbar />
      <main className="relative">
        <Hero />
        <LiveDemo />
        <Architecture />
        <Performance />
        <BentoStats />
      </main>
      <Footer />
    </div>
  );
}

export default App;

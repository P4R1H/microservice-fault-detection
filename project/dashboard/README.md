# Multimodal RCA Dashboard

A Vercel-style demo website for the Multimodal Root Cause Analysis model.

## Features

- 🎨 **Beautiful Dark UI** - Vercel-inspired design with glassmorphism effects
- 📊 **Interactive Charts** - Performance comparisons with Recharts
- 🏗️ **Architecture Visualization** - Animated model flow diagram
- 📈 **Ablation Study** - Component contribution analysis
- 🔬 **Dataset Explorer** - Explore OnlineBoutique, SockShop, TrainTicket
- ⚡ **Live Inference Demo** - Run actual model inference from the browser

## Quick Start

### Option 1: Full Setup (with backend)

1. **Install frontend dependencies:**
   ```bash
   cd dashboard
   npm install
   ```

2. **Install backend dependencies:**
   ```bash
   pip install fastapi uvicorn
   ```

3. **Start both servers:**
   ```bash
   # Windows:
   start.bat
   
   # Or manually:
   # Terminal 1 - Backend
   cd api && python server.py
   
   # Terminal 2 - Frontend
   npm run dev
   ```

4. Open http://localhost:5173

### Option 2: Frontend Only (Demo Mode)

If you don't have the trained models, the dashboard will run in simulation mode:

```bash
cd dashboard
npm install
npm run dev
```

## Tech Stack

- **React** + **Vite** - Fast development and builds
- **Tailwind CSS** - Utility-first styling
- **Framer Motion** - Smooth animations
- **Recharts** - Data visualization
- **Lucide React** - Beautiful icons
- **FastAPI** - Python backend for inference

## Project Structure

```
dashboard/
├── api/
│   ├── server.py          # FastAPI backend
│   └── requirements.txt
├── src/
│   ├── components/
│   │   ├── Hero.tsx
│   │   ├── PerformanceChart.tsx
│   │   ├── Architecture.tsx
│   │   ├── AblationChart.tsx
│   │   ├── DataExplorer.tsx
│   │   ├── LiveDemo.tsx
│   │   ├── Navigation.tsx
│   │   └── ui/
│   ├── data/
│   │   └── data.ts        # All mock data constants
│   ├── App.tsx
│   ├── index.css
│   └── main.tsx
├── index.html
├── start.bat
└── package.json
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Check if models are loaded |
| `/api/cases` | GET | List all test cases |
| `/api/case/{id}` | GET | Get case details |
| `/api/inference` | POST | Run inference on a case |
| `/api/stats` | GET | Get model statistics |

## Customization

All mock data is centralized in `src/data/data.ts`. Update this file to:
- Change performance metrics
- Modify ablation study data
- Update dataset information
- Adjust architecture flow

## Authors

- Parth Gupta
- Pratyush Jain
- Vipul Kumar Chauhan

## License

MIT License - Shiv Nadar University, 2025
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...

      // Remove tseslint.configs.recommended and replace with this
      tseslint.configs.recommendedTypeChecked,
      // Alternatively, use this for stricter rules
      tseslint.configs.strictTypeChecked,
      // Optionally, add this for stylistic rules
      tseslint.configs.stylisticTypeChecked,

      // Other configs...
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```

You can also install [eslint-plugin-react-x](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-x) and [eslint-plugin-react-dom](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-dom) for React-specific lint rules:

```js
// eslint.config.js
import reactX from 'eslint-plugin-react-x'
import reactDom from 'eslint-plugin-react-dom'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...
      // Enable lint rules for React
      reactX.configs['recommended-typescript'],
      // Enable lint rules for React DOM
      reactDom.configs.recommended,
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```

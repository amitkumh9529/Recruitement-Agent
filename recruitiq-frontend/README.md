# RecruitIQ — Frontend

React + Vite frontend for the RecruitIQ AI Resume Intelligence platform.

## Quick Start

```bash
# 1. Install dependencies
npm install

# 2. Start dev server (proxies /api → http://localhost:5000)
npm run dev

# 3. Open browser
open http://localhost:3000
```

Make sure the Flask backend (`app.py`) is running on port 5000 first.

## Build for Production

```bash
npm run build      # outputs to /dist
npm run preview    # preview the built app
```

## Project Structure

```
recruitiq-frontend/
├── index.html                  # HTML entry point
├── vite.config.js              # Vite config + dev proxy
├── package.json
├── .env.example
└── src/
    ├── main.jsx                # React entry, mounts <App>
    ├── App.jsx                 # Root layout, tab routing, top-level state
    ├── App.module.css
    │
    ├── styles/
    │   └── global.css          # CSS variables, resets, animations
    │
    ├── services/
    │   └── api.js              # All fetch/axios calls to Flask backend
    │
    ├── hooks/
    │   ├── useSession.js       # Session ID lifecycle + cleanup
    │   ├── useAlert.js         # Toast/alert state with auto-dismiss
    │   └── useAnalysis.js      # Core analysis state + runAnalysis()
    │
    ├── utils/
    │   └── helpers.js          # Pure utility functions
    │
    └── components/
        ├── layout/
        │   ├── Header.jsx / .module.css
        │   └── Sidebar.jsx / .module.css   # Config form, uploads, trigger
        │
        ├── tabs/
        │   ├── ResultsTab.jsx / .module.css   # Score ring, skill bars, tags
        │   ├── QATab.jsx / .module.css         # Chat-style RAG Q&A
        │   ├── InterviewTab.jsx / .module.css  # Question generator
        │   ├── ImproveTab.jsx / .module.css    # Improvement suggestions
        │   └── RewriteTab.jsx / .module.css    # AI resume rewrite
        │
        └── ui/                              # Reusable primitive components
            ├── Button.jsx / .module.css
            ├── Alert.jsx / .module.css
            ├── UploadZone.jsx / .module.css
            ├── ScoreRing.jsx / .module.css
            ├── SkillBar.jsx / .module.css
            ├── Accordion.jsx / .module.css
            └── EmptyState.jsx / .module.css
```

import { useState } from 'react'
import Header   from './components/layout/Header'
import Sidebar  from './components/layout/Sidebar'
import ResultsTab   from './components/tabs/ResultsTab'
import QATab        from './components/tabs/QATab'
import InterviewTab from './components/tabs/InterviewTab'
import ImproveTab   from './components/tabs/ImproveTab'
import RewriteTab   from './components/tabs/RewriteTab'
import { useSession }  from './hooks/useSession'
import { useAlert }    from './hooks/useAlert'
import { useAnalysis } from './hooks/useAnalysis'
import styles from './App.module.css'

const TABS = [
  { key: 'results',   label: 'Results'       },
  { key: 'qa',        label: 'Q & A'         },
  { key: 'interview', label: 'Interview Prep' },
  { key: 'improve',   label: 'Improvement'   },
  { key: 'rewrite',   label: 'Rewrite'       },
]

export default function App() {
  const sessionId              = useSession()
  const { alert, showAlert }   = useAlert()
  const { results, analysed, loading, runAnalysis } = useAnalysis(sessionId, showAlert)
  const [activeTab, setActive] = useState('results')
  const [apiKey, setApiKey]    = useState('')

  // Keep apiKey in App so tabs can use it without prop-drilling through Sidebar
  const handleAnalyse = (formData) => {
    setApiKey(formData.apiKey)
    runAnalysis(formData)
  }

  return (
    <div className={styles.app}>
      <Header />

      <div className={styles.layout}>
        <Sidebar
          onAnalyse={handleAnalyse}
          loading={loading}
          sessionId={sessionId}
          alert={alert}
        />

        <main className={styles.main}>
          {/* Tab Navigation */}
          <nav className={styles.tabs}>
            {TABS.map(t => (
              <button
                key={t.key}
                className={`${styles.tab} ${activeTab === t.key ? styles.active : ''}`}
                onClick={() => setActive(t.key)}
              >
                {t.label}
              </button>
            ))}
          </nav>

          {/* Loading progress bar */}
          {loading && (
            <div className={styles.progressBar}>
              <div className={styles.progressFill} />
            </div>
          )}

          {/* Tab Content */}
          <div className={styles.content}>
            {activeTab === 'results'   && <ResultsTab results={results} />}
            {activeTab === 'qa'        && <QATab analysed={analysed} apiKey={apiKey} sessionId={sessionId} />}
            {activeTab === 'interview' && <InterviewTab analysed={analysed} apiKey={apiKey} sessionId={sessionId} />}
            {activeTab === 'improve'   && <ImproveTab analysed={analysed} apiKey={apiKey} sessionId={sessionId} />}
            {activeTab === 'rewrite'   && <RewriteTab analysed={analysed} apiKey={apiKey} sessionId={sessionId} />}
          </div>
        </main>
      </div>
    </div>
  )
}

import { useState } from 'react'
import { rewriteResume, getDownloadUrl } from '../../services/api'
import Button from '../ui/Button'
import Alert from '../ui/Alert'
import EmptyState from '../ui/EmptyState'
import styles from './RewriteTab.module.css'

export default function RewriteTab({ analysed, apiKey, sessionId }) {
  const [role, setRole]       = useState('')
  const [skills, setSkills]   = useState('')
  const [output, setOutput]   = useState('')
  const [loading, setLoading] = useState(false)
  const [alert, setAlert]     = useState({ message: '', visible: false })

  if (!analysed) {
    return <EmptyState icon="✍️" title="Analyse First" description="Analyse a resume to generate a fully rewritten, ATS-optimised version." />
  }

  const run = async () => {
    setLoading(true); setOutput(''); setAlert({ message: '', visible: false })
    try {
      const result = await rewriteResume({ apiKey, sessionId, targetRole: role, highlightSkills: skills })
      setOutput(result)
    } catch (e) {
      setAlert({ message: e.message, type: 'error', visible: true })
    }
    setLoading(false)
  }

  return (
    <div className={`${styles.wrap} fade-up`}>
      <div className={styles.card}>
        <h3 className={styles.cardTitle}>Rewrite Settings</h3>
        <div className={styles.row2}>
          <div className={styles.field}>
            <label className={styles.label}>Target Role</label>
            <input className={styles.input} value={role} onChange={e => setRole(e.target.value)} placeholder="e.g. ML Engineer" />
          </div>
        </div>
        <div className={styles.field} style={{ marginBottom: 16 }}>
          <label className={styles.label}>Skills to Highlight (comma-sep or full JD text)</label>
          <textarea className={`${styles.input} ${styles.textarea}`} value={skills} onChange={e => setSkills(e.target.value)}
            placeholder="Python, TensorFlow, AWS, MLOps…" />
        </div>
        {alert.visible && <Alert {...alert} />}
        <Button variant="primary" loading={loading} onClick={run}>
          {!loading && '⟳'} Generate Improved Resume
        </Button>
      </div>

      {output && (
        <div className="fade-up">
          <div className={styles.outputHeader}>
            <span className={styles.outputLabel}>Improved Resume</span>
            <Button variant="ghost" size="sm" onClick={() => window.open(getDownloadUrl(sessionId), '_blank')}>
              ↓ Download .txt
            </Button>
          </div>
          <div className={styles.output}>{output}</div>
        </div>
      )}
    </div>
  )
}

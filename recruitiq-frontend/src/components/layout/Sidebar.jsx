import { useState } from 'react'
import Button from '../ui/Button'
import Alert from '../ui/Alert'
import UploadZone from '../ui/UploadZone'
import styles from './Sidebar.module.css'

export default function Sidebar({ onAnalyse, loading, sessionId, alert }) {
  
  const [cutoff, setCutoff]       = useState(75)
  const [resumeFile, setResume]   = useState(null)
  const [jdFile, setJd]           = useState(null)
  const [skills, setSkills]       = useState('')
  

  const handleSubmit = () => onAnalyse({ cutoff, resumeFile, jdFile, roleRequirements: skills })

  return (
    <aside className={styles.sidebar}>

      {/* Config */}
      <section>
        <p className={styles.sec}>Configuration</p>
        
        <div className={styles.field} style={{ marginTop: 10 }}>
          <label className={styles.label}>Session ID</label>
          <input className={styles.input} value={sessionId} readOnly
            style={{ color: 'var(--text-dim)', cursor: 'default' }}/>
        </div>
      </section>

      <div className={styles.divider} />

      {/* Resume upload */}
      <section>
        <p className={styles.sec}>Resume</p>
        <UploadZone
          label="PDF or TXT file"
          accept=".pdf,.txt"
          file={resumeFile}
          onChange={setResume}
          icon="📄"
        />
      </section>

      {/* JD upload */}
      <section>
        <p className={styles.sec}>
          Job Description <span className={styles.opt}>(optional)</span>
        </p>
        <UploadZone
          label="PDF or TXT file"
          accept=".pdf,.txt"
          file={jdFile}
          onChange={setJd}
          icon="📋"
        />
        <div className={styles.field} style={{ marginTop: 10 }}>
          <label className={styles.label}>Or paste required skills</label>
          <input
            className={styles.input}
            value={skills}
            onChange={e => setSkills(e.target.value)}
            placeholder="Python, React, SQL, Docker…"
          />
        </div>
      </section>

      {/* Cutoff slider */}
      <section>
        <p className={styles.sec}>Selection Threshold</p>
        <div className={styles.field}>
          <label className={styles.label}>
            Cutoff Score: <span style={{ color: 'var(--accent)' }}>{cutoff}</span>
          </label>
          <input
            type="range" min={50} max={95} step={5} value={cutoff}
            onChange={e => setCutoff(+e.target.value)}
            className={styles.range}
          />
          <div className={styles.rangeRow}><span>50</span><span>95</span></div>
        </div>
      </section>

      <div className={styles.divider} />

      {alert.visible && <Alert {...alert} />}

      <Button variant="primary" fullWidth loading={loading} onClick={handleSubmit}>
        {!loading && <span>▶</span>} Analyse Resume
      </Button>

    </aside>
  )
}

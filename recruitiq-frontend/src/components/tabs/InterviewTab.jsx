import { useState } from 'react'
import { generateInterviewQuestions } from '../../services/api'
import Button from '../ui/Button'
import Alert from '../ui/Alert'
import EmptyState from '../ui/EmptyState'
import styles from './InterviewTab.module.css'

const TYPES = ['Technical', 'Behavioral', 'Coding', 'Situational']

export default function InterviewTab({ analysed, apiKey, sessionId }) {
  const [difficulty, setDifficulty]   = useState('Medium')
  const [numQ, setNumQ]               = useState(5)
  const [types, setTypes]             = useState({ Technical: true, Behavioral: true, Coding: false, Situational: false })
  const [questions, setQuestions]     = useState([])
  const [loading, setLoading]         = useState(false)
  const [alert, setAlert]             = useState({ message: '', type: 'error', visible: false })

  if (!analysed) {
    return <EmptyState icon="🎤" title="Analyse First" description="Analyse a resume to generate tailored interview questions." />
  }

  const toggle = t => setTypes(p => ({ ...p, [t]: !p[t] }))

  const generate = async () => {
    const chosen = Object.entries(types).filter(([, v]) => v).map(([k]) => k)
    if (!chosen.length) {
      setAlert({ message: 'Select at least one question type.', type: 'error', visible: true })
      return
    }
    setLoading(true); setQuestions([])
    setAlert({ message: '', visible: false })
    try {
      const qs = await generateInterviewQuestions({ apiKey, sessionId, questionTypes: chosen, difficulty, numQuestions: numQ })
      setQuestions(qs || [])
    } catch (e) {
      setAlert({ message: e.message, type: 'error', visible: true })
    }
    setLoading(false)
  }

  return (
    <div className={`${styles.wrap} fade-up`}>
      <div className={styles.card}>
        <h3 className={styles.cardTitle}>Configure Questions</h3>

        <div className={styles.row2}>
          <div className={styles.field}>
            <label className={styles.label}>Difficulty</label>
            <select className={styles.select} value={difficulty} onChange={e => setDifficulty(e.target.value)}>
              {['Easy', 'Medium', 'Hard'].map(d => <option key={d}>{d}</option>)}
            </select>
          </div>
          <div className={styles.field}>
            <label className={styles.label}>Count</label>
            <select className={styles.select} value={numQ} onChange={e => setNumQ(+e.target.value)}>
              {[3, 5, 8, 10].map(n => <option key={n}>{n}</option>)}
            </select>
          </div>
        </div>

        <div className={styles.field} style={{ marginBottom: 16 }}>
          <label className={styles.label}>Question Types</label>
          <div className={styles.checks}>
            {TYPES.map(t => (
              <label key={t} className={styles.chk}>
                <input type="checkbox" checked={!!types[t]} onChange={() => toggle(t)} /> {t}
              </label>
            ))}
          </div>
        </div>

        {alert.visible && <Alert {...alert} />}
        <Button variant="primary" loading={loading} onClick={generate}>
          {!loading && '⚡'} Generate Questions
        </Button>
      </div>

      {questions.map((q, i) => (
        <div key={i} className={styles.qcard} style={{ animationDelay: `${i * 0.06}s` }}>
          <div className={styles.qtype}>{q.type}</div>
          <div className={styles.qtext}>{q.question}</div>
        </div>
      ))}
    </div>
  )
}

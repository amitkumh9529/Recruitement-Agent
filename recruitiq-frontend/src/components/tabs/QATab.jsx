import { useState, useRef, useEffect } from 'react'
import { askQuestion } from '../../services/api'
import Button from '../ui/Button'
import EmptyState from '../ui/EmptyState'
import styles from './QATab.module.css'

export default function QATab({ analysed, apiKey, sessionId }) {
  const [messages, setMessages] = useState([])
  const [input, setInput]       = useState('')
  const [loading, setLoading]   = useState(false)
  const bottomRef = useRef()

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  if (!analysed) {
    return (
      <EmptyState icon="💬" title="Analyse First"
        description="Analyse a resume to ask questions about the candidate." />
    )
  }

  const send = async () => {
    const q = input.trim()
    if (!q || loading) return
    setMessages(m => [...m, { role: 'user', text: q }])
    setInput('')
    setLoading(true)
    try {
      const answer = await askQuestion({ apiKey, sessionId, question: q })
      setMessages(m => [...m, { role: 'ai', text: answer }])
    } catch (e) {
      setMessages(m => [...m, { role: 'ai', text: `Error: ${e.message}` }])
    }
    setLoading(false)
  }

  return (
    <div className={`${styles.wrap} fade-up`}>
      <div className={styles.window}>
        {messages.length === 0 && (
          <div className={styles.placeholder}>Ask anything about the candidate's resume…</div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`${styles.msg} ${m.role === 'user' ? styles.msgUser : styles.msgAi}`}>
            <div className={styles.label}>{m.role === 'user' ? 'You' : 'RecruitIQ'}</div>
            <div className={`${styles.bubble} ${m.role === 'user' ? styles.bubbleUser : styles.bubbleAi}`}>
              {m.text}
            </div>
          </div>
        ))}
        {loading && (
          <div className={`${styles.msg} ${styles.msgAi}`}>
            <div className={styles.label}>RecruitIQ</div>
            <div className={`${styles.bubble} ${styles.bubbleAi} ${styles.thinking}`}>
              <span /><span /><span />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className={styles.inputRow}>
        <input
          className={styles.chatInput}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && send()}
          placeholder="Ask about skills, experience, projects…"
        />
        <Button variant="primary" size="sm" loading={loading} onClick={send}>
          {!loading && '→ Send'}
        </Button>
      </div>
    </div>
  )
}

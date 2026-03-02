import { useState } from 'react'
import { getImprovements } from '../../services/api'
import { flattenImprovements } from '../../utils/helpers'
import Button from '../ui/Button'
import Alert from '../ui/Alert'
import Accordion from '../ui/Accordion'
import EmptyState from '../ui/EmptyState'
import styles from './ImproveTab.module.css'

const AREAS = ['Skills Highlighting', 'Work Experience', 'Education', 'Summary/Objective', 'Formatting', 'ATS Optimization']

export default function ImproveTab({ analysed, apiKey, sessionId }) {
  const [selected, setSelected] = useState({ 'Skills Highlighting': true, 'Work Experience': true, 'ATS Optimization': true })
  const [targetRole, setRole]   = useState('')
  const [items, setItems]       = useState([])
  const [loading, setLoading]   = useState(false)
  const [alert, setAlert]       = useState({ message: '', visible: false })

  if (!analysed) {
    return <EmptyState icon="✨" title="Analyse First" description="Analyse a resume to get actionable improvement suggestions." />
  }

  const toggle = a => setSelected(p => ({ ...p, [a]: !p[a] }))

  const get = async () => {
    const areas = Object.entries(selected).filter(([, v]) => v).map(([k]) => k)
    if (!areas.length) { setAlert({ message: 'Select at least one area.', visible: true }); return }
    setLoading(true); setItems([])
    try {
      const data = await getImprovements({ apiKey, sessionId, improvementAreas: areas, targetRole })
      setItems(flattenImprovements(data))
    } catch (e) {
      setAlert({ message: e.message, type: 'error', visible: true })
    }
    setLoading(false)
  }

  return (
    <div className={`${styles.wrap} fade-up`}>
      <div className={styles.card}>
        <h3 className={styles.cardTitle}>Improvement Areas</h3>
        <div className={styles.checks}>
          {AREAS.map(a => (
            <label key={a} className={styles.chk}>
              <input type="checkbox" checked={!!selected[a]} onChange={() => toggle(a)} /> {a}
            </label>
          ))}
        </div>
        <div className={styles.field} style={{ marginBottom: 16, marginTop: 14 }}>
          <label className={styles.label}>Target Role</label>
          <input className={styles.input} value={targetRole} onChange={e => setRole(e.target.value)}
            placeholder="e.g. Senior Backend Engineer" />
        </div>
        {alert.visible && <Alert {...alert} />}
        <Button variant="primary" loading={loading} onClick={get}>
          {!loading && '✦'} Get Suggestions
        </Button>
      </div>

      {items.map((item, i) => (
        <Accordion key={item.area} title={item.area}
          description={item.description} items={item.items}
          defaultOpen={i === 0} />
      ))}
    </div>
  )
}

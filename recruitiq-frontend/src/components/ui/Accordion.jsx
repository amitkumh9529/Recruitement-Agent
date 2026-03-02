import { useState } from 'react'
import styles from './Accordion.module.css'

export default function Accordion({ title, description, items = [], defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen)

  return (
    <div className={styles.item}>
      <button className={styles.header} onClick={() => setOpen(o => !o)}>
        <span className={styles.title}>{title}</span>
        <span className={styles.arrow} style={{ transform: open ? 'rotate(180deg)' : 'none' }}>▼</span>
      </button>
      {open && (
        <div className={styles.body}>
          {description && <p className={styles.desc}>{description}</p>}
          {items.length > 0 && (
            <ul className={styles.list}>
              {items.map((item, i) => <li key={i}>{item}</li>)}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}

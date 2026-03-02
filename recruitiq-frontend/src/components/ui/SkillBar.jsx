import { useState, useEffect } from 'react'
import { skillClass } from '../../utils/helpers'
import styles from './SkillBar.module.css'

export default function SkillBar({ skill, score10 = 0 }) {
  const [width, setWidth] = useState(0)
  const pct = (score10 / 10) * 100
  const cls = skillClass(score10)

  useEffect(() => {
    const t = setTimeout(() => setWidth(pct), 150)
    return () => clearTimeout(t)
  }, [pct])

  return (
    <div className={styles.row}>
      <span className={styles.name}>{skill}</span>
      <div className={styles.track}>
        <div
          className={`${styles.fill} ${styles[cls]}`}
          style={{ width: `${width}%` }}
        />
      </div>
      <span className={styles.val}>{score10}/10</span>
    </div>
  )
}

import { useState, useEffect } from 'react'
import styles from './ScoreRing.module.css'

const R = 45
const CIRC = 2 * Math.PI * R

export default function ScoreRing({ score = 0, pass = false, size = 100 }) {
  const [displayed, setDisplayed] = useState(0)
  const offset = CIRC - (score / 100) * CIRC
  const color = pass ? 'var(--green)' : 'var(--red)'

  useEffect(() => {
    setDisplayed(0)
    const step = score / 60
    let cur = 0
    const t = setInterval(() => {
      cur = Math.min(cur + step, score)
      setDisplayed(Math.round(cur))
      if (cur >= score) clearInterval(t)
    }, 16)
    return () => clearInterval(t)
  }, [score])

  return (
    <div className={styles.ring} style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox="0 0 100 100" style={{ transform: 'rotate(-90deg)' }}>
        <circle className={styles.track} cx="50" cy="50" r={R} />
        <circle
          className={styles.fill}
          cx="50" cy="50" r={R}
          style={{ stroke: color, strokeDashoffset: offset }}
        />
      </svg>
      <div className={styles.num}>{displayed}</div>
    </div>
  )
}

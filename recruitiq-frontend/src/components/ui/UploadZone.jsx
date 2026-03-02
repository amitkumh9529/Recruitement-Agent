import { useState, useRef } from 'react'
import styles from './UploadZone.module.css'

export default function UploadZone({ label, accept = '.pdf,.txt', file, onChange, icon = '📄' }) {
  const [drag, setDrag] = useState(false)
  const inputRef = useRef()

  const handleDragOver = (e) => { e.preventDefault(); setDrag(true) }
  const handleDragLeave = () => setDrag(false)
  const handleDrop = (e) => {
    e.preventDefault()
    setDrag(false)
    const f = e.dataTransfer.files[0]
    if (f) onChange(f)
  }

  return (
    <div
      className={`${styles.zone} ${drag ? styles.drag : ''}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={() => inputRef.current.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        style={{ display: 'none' }}
        onChange={(e) => onChange(e.target.files[0] || null)}
      />
      <span className={styles.icon}>{icon}</span>
      {file ? (
        <div className={styles.fileName}>{file.name}</div>
      ) : (
        <p className={styles.hint}>
          {label}<br />
          <span className={styles.cta}>click or drag & drop</span>
        </p>
      )}
    </div>
  )
}

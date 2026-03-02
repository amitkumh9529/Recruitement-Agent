import styles from './Alert.module.css'

export default function Alert({ message, type = 'error', visible }) {
  if (!visible || !message) return null
  return (
    <div className={`${styles.alert} ${styles[type]}`}>
      <span className={styles.icon}>{type === 'success' ? '✓' : '!'}</span>
      {message}
    </div>
  )
}

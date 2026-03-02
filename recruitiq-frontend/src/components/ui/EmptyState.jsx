import styles from './EmptyState.module.css'

export default function EmptyState({ icon = '🔬', title, description }) {
  return (
    <div className={styles.wrap}>
      <div className={styles.icon}>{icon}</div>
      <h3 className={styles.title}>{title}</h3>
      <p className={styles.desc}>{description}</p>
    </div>
  )
}

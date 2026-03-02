import styles from './Header.module.css'

export default function Header() {
  return (
    <header className={styles.hdr}>
      <div className={styles.logo}>
        <div className={styles.dot} />
        RecruitIQ
      </div>
      <span className={styles.badge}>AI Resume Intelligence</span>
    </header>
  )
}

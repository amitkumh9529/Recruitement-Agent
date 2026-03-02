import ScoreRing from '../ui/ScoreRing'
import SkillBar from '../ui/SkillBar'
import EmptyState from '../ui/EmptyState'
import styles from './ResultsTab.module.css'

export default function ResultsTab({ results }) {
  if (!results) {
    return (
      <EmptyState
        icon="🔬"
        title="No Analysis Yet"
        description="Upload a resume and click Analyse Resume to see the full breakdown here."
      />
    )
  }

  const {
    overall_score: score = 0,
    selected,
    skill_scores = {},
    strengths = [],
    missing_skills = [],
    reasoning = '',
    detailed_weaknesses = [],
  } = results

  return (
    <div className={`${styles.wrap} fade-up`}>

      {/* Score Hero */}
      <div className={styles.hero}>
        <ScoreRing score={score} pass={selected} size={100} />
        <div className={styles.heroInfo}>
          <h2 className={styles.heroTitle}>Overall Match: {score}%</h2>
          <p className={styles.heroReason}>{reasoning}</p>
          <div className={`${styles.badge} ${selected ? styles.pass : styles.fail}`}>
            <span className={styles.dot} />
            {selected ? 'Selected' : 'Not Selected'}
          </div>
        </div>
      </div>

      {/* Skill Proficiency */}
      <div className={styles.card}>
        <h3 className={styles.cardTitle}>Skill Proficiency</h3>
        {Object.entries(skill_scores).map(([skill, s]) => (
          <SkillBar key={skill} skill={skill} score10={s} />
        ))}
      </div>

      {/* Strengths + Gaps */}
      <div className={styles.twoCol}>
        <div className={styles.card}>
          <h3 className={styles.cardTitle}>✦ Strengths</h3>
          <div className={styles.tags}>
            {strengths.length
              ? strengths.map(s => <span key={s} className={`${styles.tag} ${styles.tagGreen}`}>{s}</span>)
              : <span className={styles.none}>None identified</span>}
          </div>
        </div>
        <div className={styles.card}>
          <h3 className={styles.cardTitle}>⚠ Skill Gaps</h3>
          <div className={styles.tags}>
            {missing_skills.length
              ? missing_skills.map(s => <span key={s} className={`${styles.tag} ${styles.tagRed}`}>{s}</span>)
              : <span className={styles.none}>None identified</span>}
          </div>
        </div>
      </div>

      {/* Weakness Detail */}
      {detailed_weaknesses.length > 0 && (
        <div className={styles.card} style={{ marginTop: 14 }}>
          <h3 className={styles.cardTitle}>Weakness Analysis</h3>
          {detailed_weaknesses.map((w, i) => (
            <div key={i} className={styles.wcard}>
              <div className={styles.wskill}>{w.skill}</div>
              <p className={styles.wdesc}>{w.details}</p>
              {w.improvement_suggestions?.length > 0 && (
                <ul className={styles.wlist}>
                  {w.improvement_suggestions.map((s, j) => <li key={j}>{s}</li>)}
                </ul>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

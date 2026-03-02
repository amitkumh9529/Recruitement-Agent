/** Generate a unique session ID */
export const generateSessionId = () =>
  'sess_' + Math.random().toString(36).slice(2, 10)

/** Clamp a number between min and max */
export const clamp = (val, min, max) => Math.min(Math.max(val, min), max)

/** Map a 0-10 skill score to a CSS class name */
export const skillClass = (score10) => {
  const pct = (score10 / 10) * 100
  if (pct >= 70) return 'strong'
  if (pct >= 40) return 'mid'
  return 'weak'
}

/** Format an object of improvements into a flat renderable array */
export const flattenImprovements = (improvements = {}) =>
  Object.entries(improvements).map(([area, detail]) => ({
    area,
    description: detail.description || '',
    items: detail.specific || detail.specific_suggestions || [],
  }))

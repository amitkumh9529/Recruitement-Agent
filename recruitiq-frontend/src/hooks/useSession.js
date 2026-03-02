import { useState, useEffect, useCallback } from 'react'
import { generateSessionId } from '../utils/helpers'
import { cleanupSession } from '../services/api'

/**
 * Manages a persistent session ID and cleanup on unmount / page close.
 */
export function useSession() {
  const [sessionId] = useState(generateSessionId)

  // Cleanup when the tab/window closes
  useEffect(() => {
    const handleUnload = () => cleanupSession(sessionId)
    window.addEventListener('beforeunload', handleUnload)
    return () => {
      window.removeEventListener('beforeunload', handleUnload)
      cleanupSession(sessionId)
    }
  }, [sessionId])

  return sessionId
}

import { useState, useCallback } from 'react'

/**
 * Simple alert state manager with auto-dismiss.
 * Returns { alert, showAlert }
 */
export function useAlert(timeout = 5000) {
  const [alert, setAlert] = useState({ message: '', type: 'error', visible: false })

  const showAlert = useCallback((message, type = 'error') => {
    setAlert({ message, type, visible: true })
    setTimeout(() => setAlert(a => ({ ...a, visible: false })), timeout)
  }, [timeout])

  return { alert, showAlert }
}

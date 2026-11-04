import { useEffect, useState } from 'react'

/** Minimal hash router: "#/", "#/jobs/<id>", "#/showcase/<index>". Works on any static host. */
export type Route =
  | { name: 'home' }
  | { name: 'job'; id: string }
  | { name: 'showcase'; index: number }

export function parseRoute(hash: string): Route {
  const [, section, arg] = hash.replace(/^#/, '').split('/')
  if (section === 'jobs' && arg) return { name: 'job', id: decodeURIComponent(arg) }
  if (section === 'showcase' && arg !== undefined && /^\d+$/.test(arg)) return { name: 'showcase', index: Number(arg) }
  return { name: 'home' }
}

export const hrefs = {
  home: '#/',
  job: (id: string) => `#/jobs/${encodeURIComponent(id)}`,
  showcase: (index: number) => `#/showcase/${index}`,
}

export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(() => parseRoute(window.location.hash))
  useEffect(() => {
    const onChange = () => setRoute(parseRoute(window.location.hash))
    window.addEventListener('hashchange', onChange)
    return () => window.removeEventListener('hashchange', onChange)
  }, [])
  return route
}

export function navigate(href: string): void {
  window.location.hash = href
}

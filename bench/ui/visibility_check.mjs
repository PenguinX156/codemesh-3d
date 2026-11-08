// Checks that a hidden tab stops polling the backend (so the hosted server can scale to zero).
// Usage: node bench/ui/visibility_check.mjs [base-url]
// Forces the worst case: backend health failing (offline re-checks every 10 s) and a job that never
// finishes (job poll every 1 s), then hides the tab and counts requests.
import { chromium } from 'playwright'

const base = process.argv[2] ?? 'https://fjziemski--codemesh-web.us-east.modal.run/'
const JOB = '35158431c90b4133b225c508ece7d94d'
const b = await chromium.launch()
const p = await b.newPage()
const hits = { health: 0, job: 0 }
await p.route('**/api/health', (r) => { hits.health++; return r.fulfill({ status: 503, body: 'down' }) })
await p.route(`**/api/jobs/${JOB}`, async (r) => {
  hits.job++
  const j = await (await r.fetch()).json()
  await r.fulfill({ json: { ...j, status: 'reconstructing', progress: 0.5 } })
})
const setHidden = (hidden) => p.evaluate((h) => {
  Object.defineProperty(document, 'hidden', { configurable: true, get: () => h })
  Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => (h ? 'hidden' : 'visible') })
  document.dispatchEvent(new Event('visibilitychange'))
}, hidden)
const count = async (ms) => { const s = { ...hits }; await p.waitForTimeout(ms); return { health: hits.health - s.health, job: hits.job - s.job } }

await p.goto(`${base}#/jobs/${JOB}`)
const visible = await count(12_000)
await setHidden(true)
await p.waitForTimeout(1_500) // let an in-flight request finish
const hidden = await count(15_000)
await setHidden(false)
const shown = await count(5_000)
await b.close()

console.log(JSON.stringify({ visible, hidden, shown }))
const ok = visible.job >= 5 && visible.health >= 1 && hidden.job === 0 && hidden.health === 0 && shown.job >= 1 && shown.health >= 1
console.log(ok ? 'PASS: hidden tab makes no requests and resumes when shown' : 'FAIL')
process.exit(ok ? 0 : 1)

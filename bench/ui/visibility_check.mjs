// Checks that an idle or hidden tab stops polling the backend, so the hosted server can scale to zero.
// Usage: node bench/ui/visibility_check.mjs [base-url]      (exit code 0 = all checks pass)
// Job statuses and backend health are faked with request routing, so no real job is started.
import { chromium } from 'playwright'

const base = process.argv[2] ?? 'https://fjziemski--codemesh-web.us-east.modal.run/'
const JOB = '35158431c90b4133b225c508ece7d94d' // an existing finished job on the hosted app
const b = await chromium.launch()

async function scenario({ healthUp, status }) {
  const p = await b.newPage()
  const hits = { health: 0, job: 0 }
  await p.route('**/api/health', (r) => {
    hits.health++
    return healthUp ? r.fallback() : r.fulfill({ status: 503, body: 'down' })
  })
  await p.route(`**/api/jobs/${JOB}`, async (r) => {
    hits.job++
    const j = await (await r.fetch()).json()
    await r.fulfill({ json: { ...j, status, progress: 0.5 } })
  })
  const setHidden = (hidden) => p.evaluate((h) => {
    Object.defineProperty(document, 'hidden', { configurable: true, get: () => h })
    Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => (h ? 'hidden' : 'visible') })
    document.dispatchEvent(new Event('visibilitychange'))
  }, hidden)
  const count = async (ms) => {
    const s = { ...hits }
    await p.waitForTimeout(ms)
    return { health: hits.health - s.health, job: hits.job - s.job }
  }
  await p.goto(`${base}#/jobs/${JOB}`)
  const visible = await count(12_000)
  await setHidden(true)
  await p.waitForTimeout(1_500) // let an in-flight request finish
  const hidden = await count(15_000)
  await setHidden(false)
  const shown = await count(5_000)
  await p.close()
  return { visible, hidden, shown }
}

const checks = []
const check = (name, ok, data) => { checks.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${JSON.stringify(data)}`) }

// 1. Backend down + job running: polls while visible, silent while hidden, resumes when shown.
let r = await scenario({ healthUp: false, status: 'reconstructing' })
check('running job, hidden tab is silent and resumes', r.visible.job >= 5 && r.hidden.job === 0 && r.hidden.health === 0 && r.shown.job >= 1 && r.shown.health >= 1, r)
// 2. Job waiting for the user (object selection), tab visible: one fetch, then no more requests.
r = await scenario({ healthUp: true, status: 'ready_to_select' })
check('waiting job, visible idle tab stops polling', r.visible.job <= 2 && r.visible.health <= 1, r)
// 3. Finished job, tab visible: no polling at all after load.
r = await scenario({ healthUp: true, status: 'done' })
check('finished job, visible idle tab stops polling', r.visible.job <= 2 && r.visible.health <= 1 && r.shown.job <= 1, r)

await b.close()
process.exit(checks.every(Boolean) ? 0 : 1)

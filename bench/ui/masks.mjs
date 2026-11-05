// Mocks only the job JSON of a real job (status ready_to_reconstruct + masks_preview) to check the overlay,
// then submits Reconstruct to the REAL backend to verify the request schema. Usage: node masks.mjs <jobId>
import { chromium } from 'playwright'
import fs from 'node:fs'
const id = process.argv[2]
const b = await chromium.launch()
for (const [n, v] of [['desktop', { width: 1440, height: 900 }], ['mobile', { width: 390, height: 844 }]]) {
  const p = await (await b.newContext({ viewport: v })).newPage()
  const reqs = []
  p.on('request', (r) => r.method() === 'POST' && reqs.push(r.url() + ' ' + r.postData()))
  let mock = true
  await p.route('**/api/jobs/' + id, async (r) => {
    const real = await (await r.fetch()).json()
    r.fulfill({ json: mock ? { ...real, status: 'ready_to_reconstruct', error: null, artifacts: { ...real.artifacts, masks_preview: 'masks/preview.png' } } : real })
  })
  await p.route('**/artifacts/masks/preview.png', (r) => r.fulfill({ body: fs.readFileSync('mask_fixture.png'), contentType: 'image/png' }))
  await p.goto('http://localhost:5173/#/jobs/' + id)
  await p.locator('.mask-overlay').waitFor()
  await p.waitForTimeout(800)
  await p.screenshot({ path: `masks-${n}.jpg`, type: 'jpeg', quality: 75 })
  if (n === 'desktop') {
    mock = false
    await p.getByRole('button', { name: /^reconstruct/i }).click()
    await p.waitForTimeout(2500)
    console.log(reqs.join('\n'))
    console.log('status after reconstruct:', (await (await fetch('http://localhost:8000/api/jobs/' + id)).json()).message)
  }
}
await b.close()

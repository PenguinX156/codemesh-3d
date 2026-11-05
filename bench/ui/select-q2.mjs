// Screenshots of the new select + mask-review screens. The job JSON of a real ingested job is mocked to the needed
// status (frames are real). Usage: node select-q2.mjs <jobId> [port]
import { chromium } from 'playwright'
import fs from 'node:fs'
const id = process.argv[2], port = process.argv[3] || '5191'
const b = await chromium.launch()
for (const [n, v] of [['desktop', { width: 1440, height: 900 }], ['mobile', { width: 390, height: 844 }]]) {
  for (const stage of ['select', 'masks']) {
    const p = await (await b.newContext({ viewport: v })).newPage()
    await p.route('**/api/jobs/' + id, async (r) => {
      const real = await (await r.fetch()).json()
      const artifacts = stage === 'masks' ? { ...real.artifacts, masks_preview: 'masks/preview.png' } : real.artifacts
      r.fulfill({ json: { ...real, status: stage === 'masks' ? 'ready_to_reconstruct' : 'ready_to_select', error: null, artifacts } })
    })
    await p.route('**/artifacts/masks/preview.png', (r) => r.fulfill({ body: fs.readFileSync('mask_fixture.png'), contentType: 'image/png' }))
    await p.goto(`http://localhost:${port}/#/jobs/${id}`)
    await p.locator(stage === 'select' ? '.frame-stage img' : '.mask-preview').waitFor()
    await p.waitForTimeout(1500)
    await p.screenshot({ path: `${stage}-q2-${n}.jpg`, type: 'jpeg', quality: 75, fullPage: true })
    console.log(stage, n, 'scrollWidth', await p.evaluate(() => document.documentElement.scrollWidth), 'viewport', v.width)
  }
}
await b.close()

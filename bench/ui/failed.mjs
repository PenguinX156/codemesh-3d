// Re-open an existing failed job and screenshot it: node failed.mjs <jobId>
import { chromium } from 'playwright'
const b = await chromium.launch()
for (const [n, v] of [['desktop', { width: 1440, height: 900 }], ['mobile', { width: 390, height: 844 }]]) {
  const p = await (await b.newContext({ viewport: v })).newPage()
  await p.goto('http://localhost:5173/#/jobs/' + process.argv[2])
  await p.locator('.frame-stage img').waitFor()
  await p.waitForTimeout(800)
  await p.screenshot({ path: `failed-${n}.jpg`, type: 'jpeg', quality: 75 })
}
await b.close()

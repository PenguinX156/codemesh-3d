// Drives the real UI against the real backend. Usage: node flow.mjs [home|upload|select|done|all]
import { chromium } from 'playwright'
import fs from 'node:fs'
import path from 'node:path'

const BASE = 'http://localhost:5173/'
const SIZES = { desktop: { width: 1440, height: 900 }, mobile: { width: 390, height: 844 } }
const shot = (page, name, size) => page.screenshot({ path: `${name}-${size}.jpg`, type: 'jpeg', quality: 75 })
const step = process.argv[2] ?? 'all'
const browser = await chromium.launch({ args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] })
const logs = []

async function session(size, fn) {
  const ctx = await browser.newContext({ viewport: SIZES[size], deviceScaleFactor: 1, hasTouch: size === 'mobile' })
  const page = await ctx.newPage()
  page.on('console', (m) => ['error', 'warning'].includes(m.type()) && logs.push(`[${size}] ${m.type()}: ${m.text()}`))
  page.on('pageerror', (e) => logs.push(`[${size}] pageerror: ${e.message}`))
  try { await fn(page) } finally { await ctx.close() }
}

for (const size of Object.keys(SIZES)) {
  if (step === 'home' || step === 'all')
    await session(size, async (page) => {
      await page.goto(BASE)
      await page.getByText('New capture').waitFor()
      await page.waitForTimeout(800)
      await shot(page, 'home', size)
    })

  if (step === 'upload' || step === 'all')
    await session(size, async (page) => {
      await page.goto(BASE)
      await page.locator('input[type=file]').setInputFiles(path.resolve('kitchen.mp4'))
      await page.getByRole('checkbox').check()
      await shot(page, 'upload-ready', size)
      await page.getByRole('button', { name: 'Upload and start' }).click()
      await page.getByRole('heading', { name: /select the object/i }).waitFor({ timeout: 120000 })
      await page.locator('.frame-stage img').first().waitFor()
      await page.waitForFunction(() => { const i = document.querySelector('.frame-stage img'); return i && i.naturalWidth > 0 })
      await page.waitForTimeout(500)
      await shot(page, 'select-frames', size)
      const box = await page.locator('.frame-stage').boundingBox()
      await page.mouse.click(box.x + box.width * 0.5, box.y + box.height * 0.55)
      await shot(page, 'select-point', size)
      await page.getByRole('button', { name: /segment object/i }).click()
      // Either masks arrive or the job fails; record whichever comes first.
      await page.waitForFunction(() => !!document.querySelector('.error, .mask-overlay, .bar-fill.failed') || /Reconstruct/i.test(document.body.innerText), null, { timeout: 180000 }).catch(() => {})
      await page.waitForTimeout(1500)
      await shot(page, 'after-select', size)
      console.log(size, 'job url', page.url())
    })

  if (step === 'done' || step === 'all')
    await session(size, async (page) => {
      // Mock-only: browser-side route fakes a finished job so the result view is exercised with a real GLB.
      const job = {
        id: 'a'.repeat(32), created_at: Date.now() / 1000, mode: 'walkaround', status: 'done', stage: null, progress: 1, message: 'Done',
        stages: ['ingest', 'segment', 'geometry', 'fuse', 'mesh', 'texture', 'export', 'evaluate'].map((name, i) => ({ name, status: 'done', seconds: 3.2 * (i + 1), cached: false, metrics: {} })),
        artifacts: { model_glb: 'export/model.glb', compare_0: 'evaluate/compare_0.png' }, metrics: {}, error: null,
        settings: { reconstruct: { preset: 'standard' } },
      }
      job.stages[2].metrics = { engine: 'mapanything-cpu' }
      job.stages[5].metrics = { texture_size: 2048, triangles: 1280 }
      job.stages[7].metrics = { iou_holdout_mean: 0.9312, psnr_mean: 23.41, ssim_mean: 0.871 }
      const png = fs.readFileSync(path.resolve('kitchen.mp4').replace('kitchen.mp4', 'compare_fixture.png'))
      await page.route('**/api/jobs/' + job.id, (r) => r.fulfill({ json: job }))
      await page.route('**/api/jobs/' + job.id + '/artifacts/evaluate/compare_0.png', (r) => r.fulfill({ body: png, contentType: 'image/png' }))
      await page.route('**/api/jobs/' + job.id + '/artifacts/export/model.glb', (r) => r.fulfill({ body: fs.readFileSync(path.resolve('../../frontend/public/showcase/sample.glb')), contentType: 'model/gltf-binary' }))
      await page.goto(BASE + '#/jobs/' + job.id)
      await page.getByText('Stage timings').waitFor()
      await page.locator('canvas').waitFor({ timeout: 30000 })
      await page.waitForTimeout(4000)
      await shot(page, 'result', size)
      const err = await page.locator('.viewer [role=alert]').count()
      console.log(size, 'viewer errors', err, 'scale note:', await page.getByTestId('scale-note').innerText())
    })

  if (step === 'showcase' || step === 'all')
    await session(size, async (page) => {
      await page.goto(BASE + '#/showcase/0')
      await page.locator('canvas').waitFor({ timeout: 30000 })
      await page.waitForTimeout(4000)
      await shot(page, 'showcase', size)
    })
}
await browser.close()
console.log(logs.join('\n') || 'no console errors')

// Run with: npm test  (node's built-in runner; core is plain TS so no bundler is needed)
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { createApi } from './api.ts'
import { pollJob } from './jobPoller.ts'
import { parseTriangleBudget } from './presets.ts'
import { compareArtifacts, summarize } from './summary.ts'
import type { Job } from './types.ts'

const job = (status: Job['status']): Job => ({
  id: 'j1', created_at: 0, mode: 'walkaround', status, stage: null, progress: 0, message: '', stages: [], artifacts: {}, metrics: {},
})

test('resolveUrl and artifactUrl honour the injected base', () => {
  const api = createApi('http://box:8000/api/')
  assert.equal(api.resolveUrl('/api/jobs/j1/frames/0.jpg'), 'http://box:8000/api/jobs/j1/frames/0.jpg')
  assert.equal(api.resolveUrl('https://cdn/x.png'), 'https://cdn/x.png')
  assert.equal(api.artifactUrl('j1', 'model.glb'), 'http://box:8000/api/jobs/j1/artifacts/model.glb')
})

test('api surfaces FastAPI error detail', async () => {
  const fake = (async () => new Response(JSON.stringify({ detail: 'nope' }), { status: 422 })) as typeof fetch
  await assert.rejects(createApi('/api', fake).getJob('x'), { message: 'nope', status: 422 })
})

test('pollJob recovers from errors and stops on a terminal status', async () => {
  const seq = ['err', 'segmenting', 'done', 'never']
  let calls = 0
  const fake = (async () => {
    const next = seq[calls++]
    if (next === 'err') throw new Error('offline')
    return new Response(JSON.stringify(job(next as Job['status'])))
  }) as typeof fetch
  const seen: string[] = []
  await new Promise<void>((resolve) => {
    pollJob(createApi('/api', fake), 'j1', {
      intervalMs: 5,
      maxBackoffMs: 10,
      onError: () => seen.push('error'),
      onUpdate: (j) => {
        seen.push(j.status)
        if (j.status === 'done') setTimeout(resolve, 40)
      },
    })
  })
  assert.deepEqual(seen, ['error', 'segmenting', 'done'])
  assert.equal(calls, 3)
})

test('parseTriangleBudget clamps and falls back to auto', () => {
  assert.equal(parseTriangleBudget(''), 'auto')
  assert.equal(parseTriangleBudget('10'), 500)
  assert.equal(parseTriangleBudget('25000'), 25000)
  assert.equal(parseTriangleBudget('9e9'), 2_000_000)
})

test('summarize reads headline metrics from stages and the scale source from settings', () => {
  const j = {
    ...job('done'),
    stages: [
      { name: 'geometry', status: 'done' as const, seconds: 1, metrics: { engine: 'torch' } },
      { name: 'evaluate', status: 'done' as const, seconds: 1, metrics: { iou_holdout_mean: 0.9, psnr_mean: 24, triangles: 100 } },
    ],
    settings: { reconstruct: { known_dimension: { axis: 'y', meters: 0.2 } } },
    artifacts: { compare_10: 'c10.png', compare_2: 'c2.png', model_glb: 'm.glb' },
  }
  const s = summarize(j)
  assert.deepEqual([s.iou, s.psnr, s.ssim, s.triangles, s.engine, s.measuredScale], [0.9, 24, null, 100, 'torch', true])
  assert.deepEqual(compareArtifacts(j).map(([k]) => k), ['compare_2', 'compare_10'])
})

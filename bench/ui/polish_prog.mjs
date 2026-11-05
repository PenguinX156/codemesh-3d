import { chromium } from 'playwright'
const J='4f6643e1e8e3465dbbfacca032551636'
const b = await chromium.launch()
const p = await (await b.newContext({viewport:{width:1440,height:900}})).newPage()
await p.route('**/api/jobs/'+J, async r => { const j = await (await r.fetch()).json(); j.status='reconstructing'; j.progress=0.45; j.stage='mesh'; j.message='visual hull + depth carve'; let i=0; j.stages.forEach(s=>{ s.status = i<4?'done':i===4?'running':'pending'; i++ }); await r.fulfill({json:j}) })
await p.goto('http://localhost:5180/#/jobs/'+J); await p.waitForTimeout(1500)
await p.screenshot({path:'polish_after_progress_d.jpg',type:'jpeg',quality:75})
await b.close()

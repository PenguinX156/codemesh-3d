import { chromium } from 'playwright'
const [,, tag, base='http://localhost:5180/'] = process.argv
const J='4f6643e1e8e3465dbbfacca032551636'
const b = await chromium.launch({ args: ['--use-gl=swiftshader','--enable-unsafe-swiftshader','--ignore-gpu-blocklist'] })
for (const [n,s] of [['d',{width:1440,height:900}],['m',{width:390,height:844}]]) {
  const p = await (await b.newContext({viewport:s})).newPage()
  const errs=[]; p.on('pageerror',e=>errs.push(e.message)); p.on('console',m=>m.type()==='error'&&errs.push(m.text()))
  // the demo job may be mid re-run; force the done view from its (complete) stage data
  await p.route('**/api/jobs/'+J, async r => { const j = await (await r.fetch()).json(); j.status='done'; j.progress=1; j.stage=null; await r.fulfill({json:j}) })
  await p.goto(base); await p.waitForTimeout(1200)
  await p.screenshot({path:`polish_${tag}_home_${n}.jpg`,type:'jpeg',quality:75})
  await p.goto(base+`#/jobs/${J}`); await p.waitForTimeout(6000)
  await p.screenshot({path:`polish_${tag}_result_${n}.jpg`,type:'jpeg',quality:75,fullPage:n==='m'})
  console.log(n, errs.slice(0,5))
}
await b.close()

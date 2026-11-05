import { chromium } from 'playwright'
const J='4f6643e1e8e3465dbbfacca032551636'
const b = await chromium.launch({ args: ['--use-gl=swiftshader','--enable-unsafe-swiftshader'] })
const p = await (await b.newContext({viewport:{width:390,height:844}})).newPage()
await p.route('**/api/jobs/'+J, async r => { const j = await (await r.fetch()).json(); j.status='done'; j.progress=1; await r.fulfill({json:j}) })
await p.goto('http://localhost:5180/#/jobs/'+J); await p.waitForTimeout(4000)
console.log(await p.evaluate(()=>[document.documentElement.scrollWidth, ...[...document.querySelectorAll('body *')].filter(e=>e.getBoundingClientRect().right>391).slice(0,8).map(e=>e.tagName+'.'+e.className+' '+Math.round(e.getBoundingClientRect().right))]))
await p.screenshot({path:'polish_after_result_m.jpg',type:'jpeg',quality:75})
await b.close()

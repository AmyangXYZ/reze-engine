import { readFileSync, writeFileSync } from "node:fs"
import { gzipSync, brotliCompressSync, constants } from "node:zlib"
const CUBE="/Users/amyang/Projects/reze-engine/blender-5.2.0/release/datafiles/colormanagement/luts/AgX_Base_sRGB.cube"
let N=0; const data=[]
for (const raw of readFileSync(CUBE,"utf8").split("\n")) {
  const s=raw.trim(); if(!s||s.startsWith("#")||s.startsWith("TITLE")||s.startsWith("DOMAIN"))continue
  if(s.startsWith("LUT_3D_SIZE")){N=parseInt(s.split(/\s+/)[1],10);continue}
  const p=s.split(/\s+/).map(Number); if(p.length===3&&p.every(Number.isFinite))data.push(p)
}
// rgb10a2unorm: 10 bits/channel — four bits past what an 8-bit display resolves,
// which leaves headroom for the grade that runs after the transform.
const u32=new Uint32Array(N**3)
const q=(v)=>Math.max(0,Math.min(1023,Math.round(Math.max(0,Math.min(1,v))*1023)))
data.forEach((p,i)=>{ u32[i] = q(p[0]) | (q(p[1])<<10) | (q(p[2])<<20) | (3<<30) })
const bin=Buffer.from(u32.buffer)
const gz=gzipSync(bin,{level:9})
const br=brotliCompressSync(bin,{params:{[constants.BROTLI_PARAM_QUALITY]:11}})
console.log(`raw          ${(bin.length/1024).toFixed(0).padStart(5)} KB`)
console.log(`gzip         ${(gz.length/1024).toFixed(0).padStart(5)} KB   base64 ${(gz.length*4/3/1024).toFixed(0)} KB`)
console.log(`brotli       ${(br.length/1024).toFixed(0).padStart(5)} KB   base64 ${(br.length*4/3/1024).toFixed(0)} KB`)
// Quantisation error introduced by 10-bit, in 8-bit display units.
let max=0
data.forEach((p,i)=>{ const w=u32[i]
  const back=[(w&1023)/1023,((w>>10)&1023)/1023,((w>>20)&1023)/1023]
  for(let c=0;c<3;c++) max=Math.max(max,Math.abs(back[c]-Math.max(0,Math.min(1,p[c]))))
})
console.log(`\n10-bit quantisation error: ${(max*255).toFixed(4)} / 255  (i.e. ${(max*1023).toFixed(2)} of a 10-bit step)`)
writeFileSync("agx.gz", gz)

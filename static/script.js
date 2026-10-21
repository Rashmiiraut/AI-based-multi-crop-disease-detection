const input=document.getElementById("image");
const browse=document.getElementById("browse");
const drop=document.getElementById("drop");
const preview=document.getElementById("preview");
const predict=document.getElementById("predict");
const clear=document.getElementById("clear");
const status=document.getElementById("status");

browse.onclick=()=>input.click();
drop.onclick=e=>{if(e.target!==browse)input.click()};

["dragenter","dragover"].forEach(n=>drop.addEventListener(n,e=>{
  e.preventDefault();drop.style.borderColor="#287a52";drop.style.background="#eaf5ee";
}));
["dragleave","drop"].forEach(n=>drop.addEventListener(n,e=>{
  e.preventDefault();drop.style.borderColor="";drop.style.background="";
}));
drop.addEventListener("drop",e=>{
  const f=e.dataTransfer.files[0];
  if(f) handleFile(f);
});
input.onchange=()=>{if(input.files[0])handleFile(input.files[0])};

function handleFile(file){
  if(!["image/jpeg","image/png"].includes(file.type)){
    status.textContent="Please choose a JPG, JPEG, or PNG image.";return;
  }
  const r=new FileReader();
  r.onload=e=>{
    preview.src=e.target.result;preview.classList.remove("hidden");
    drop.classList.add("hidden");predict.disabled=false;clear.disabled=false;
    status.textContent="";
    document.getElementById("empty").classList.remove("hidden");
    document.getElementById("result").classList.add("hidden");
  };
  r.readAsDataURL(file);
}
clear.onclick=()=>{
  input.value="";preview.src="";preview.classList.add("hidden");
  drop.classList.remove("hidden");predict.disabled=true;clear.disabled=true;status.textContent="";
};
predict.onclick=async()=>{
  if(!input.files[0])return;
  predict.disabled=true;status.textContent="Analyzing image...";
  const fd=new FormData();fd.append("image",input.files[0]);
  try{
    const res=await fetch("/predict",{method:"POST",body:fd});
    const data=await res.json();
    if(!res.ok||!data.success)throw new Error(data.message||"Prediction failed.");
    document.getElementById("crop").textContent=data.crop;
    document.getElementById("condition").textContent=data.prediction;
    document.getElementById("confidence").textContent=data.confidence.toFixed(2)+"%";
    document.getElementById("bar").style.width=Math.min(data.confidence,100)+"%";
    const box=document.getElementById("probs");box.innerHTML="";
    Object.entries(data.probabilities).sort((a,b)=>b[1]-a[1]).forEach(([name,value])=>{
      const row=document.createElement("div");row.className="prob";
      row.innerHTML=`<span>${safe(name)}</span><b>${Number(value).toFixed(2)}%</b>`;
      box.appendChild(row);
    });
    document.getElementById("empty").classList.add("hidden");
    document.getElementById("result").classList.remove("hidden");
    document.getElementById("badge").textContent="Ready";
    status.textContent="Prediction completed.";
  }catch(e){status.textContent=e.message}
  finally{predict.disabled=false}
};
function safe(s){return s.replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[c]))}

async function loadMetrics(){
  try{
    const d=await (await fetch("/metrics")).json();
    if(!d.available)return;
    for(const [id,key] of [["accuracy","accuracy"],["precision","precision"],["recall","recall"],["f1","f1_score"]]){
      document.getElementById(id).textContent=(Number(d[key])*100).toFixed(2)+"%";
    }
  }catch(e){}
}
loadMetrics();

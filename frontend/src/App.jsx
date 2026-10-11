
import { useEffect, useState } from "react";
import axios from "axios";
import {
  Activity, BarChart3, Brain, CheckCircle, HeartPulse,
  Info, Menu, Upload, AlertTriangle
} from "lucide-react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid
} from "recharts";
import "./App.css";

const API = import.meta.env.VITE_API_URL || "/api";
const ACCEPTED = ".zip,.png,.jpg,.jpeg,.pdf";
const names = {N:"Normal",S:"Supraventricular",V:"Ventricular",F:"Fusion",Q:"Unknown / Other"};

export default function App() {
  const [page,setPage]=useState("dashboard");
  const [health,setHealth]=useState();
  const [digitizer,setDigitizer]=useState();
  const [info,setInfo]=useState();
  const [performance,setPerformance]=useState();
  const [file,setFile]=useState(null);
  const [analysis,setAnalysis]=useState(null);
  const [busy,setBusy]=useState(false);
  const [error,setError]=useState("");

  useEffect(()=>{
    let mounted = true;
    const load = async () => {
      try {
        const h = await axios.get(`${API}/health`);
        if (mounted) { setHealth(h.data); setDigitizer(h.data.image_digitizer); }
      } catch (e) {
        if (mounted) setError("FastAPI is not reachable on port 8000. Start the backend and refresh this page.");
      }
      try {
        const m = await axios.get(`${API}/model-info`);
        if (mounted) setInfo(m.data);
      } catch {}
      try {
        const p = await axios.get(`${API}/performance`);
        if (mounted) setPerformance(p.data);
      } catch {}
    };
    load();
    return () => { mounted = false; };
  },[]);

  async function analyze(){
    if(!file){setError("Choose an ECG image/PDF or a WFDB ZIP first.");return;}
    setBusy(true);setError("");setAnalysis(null);
    try{
      const form=new FormData();form.append("file",file);
      const {data}=await axios.post(`${API}/analyze-upload?max_beats=5000`,form,{
        headers:{"Content-Type":"multipart/form-data"},
        maxContentLength:100*1024*1024,
        maxBodyLength:100*1024*1024
      });
      setAnalysis(data);
    }catch(e){setError(e.response?.data?.detail||"ECG analysis failed.");}
    finally{setBusy(false);}
  }

  const nav=[
    ["dashboard","Dashboard",BarChart3],
    ["prediction","ECG Prediction",HeartPulse],
    ["analysis","Feature Analysis",Activity],
    ["performance","Model Performance",Brain],
    ["about","About",Info]
  ];

  return <div className="app">
    <aside className="side">
      <div className="brand"><div className="brandIcon"><HeartPulse/></div><div><b>SmartECG</b><small>ML</small></div></div>
      {nav.map(([id,n,I])=><button className={page===id?"nav active":"nav"} onClick={()=>setPage(id)} key={id}><I size={19}/>{n}</button>)}
      <div className="local"><CheckCircle size={16}/>Local Server</div>
    </aside>
    <main>
      <header>
        <button className="menu" aria-label="Menu"><Menu/></button>
        <div><b>SmartECG-ML</b><small>ECG Classification Platform</small></div>
        <div className="statusStack"><span className={health?.status==="healthy"?"online":"offline"}>● {health?.status==="healthy"?"Backend Online":"Backend Offline"}</span><small className={digitizer?.production_ready||digitizer?.fallback_enabled?"online":"offline"}>● {digitizer?.active_engine==="classical_fallback"?"Fast Image Digitizer Ready":digitizer?.production_ready?"Production ECG Digitizer Ready":"Image Digitizer Not Configured"}</small></div>
      </header>
      <section className="content">
        {error && <div className="globalError"><AlertTriangle size={18}/>{error}</div>}
        {page==="dashboard"&&<Dashboard go={setPage} digitizer={digitizer}/>}
        {page==="prediction"&&<Prediction file={file} setFile={setFile} analyze={analyze} analysis={analysis} busy={busy} digitizer={digitizer}/>}
        {page==="analysis"&&<Analysis features={info?.features||[]}/>}
        {page==="performance"&&<Performance data={performance}/>}
        {page==="about"&&<About/>}
      </section>
      <footer>SmartECG-ML <span>Local Research Prototype</span></footer>
    </main>
  </div>
}

function Dashboard({go,digitizer}){
  return <>
    <div className="head"><h1>SmartECG-ML</h1><p>ECG Beat Classification System</p></div>
    <div className="hero"><HeartPulse/><h2>Analyze an ECG</h2><p>{digitizer?.production_ready||digitizer?.fallback_enabled?"Upload a paper ECG image or analyze a WFDB ZIP with the existing 23-feature model.":"WFDB ZIP analysis is available. Paper ECG analysis needs an available image digitizer."}</p><button className="primary" onClick={()=>go("prediction")}>Analyze an ECG</button></div>
    <div className="grid">{[["ECG Beats","109,395"],["Patients","47"],["Features","23"],["Classes","5"]].map(x=><div className="card" key={x[0]}><small>{x[0]}</small><strong>{x[1]}</strong></div>)}</div>
    <div className="card"><h3>Nested Patient-Wise Validation</h3><div className="grid">{[["Accuracy","86.37%"],["Balanced Accuracy","59.72%"],["Macro F1","54.73%"],["Weighted F1","87.63%"]].map(x=><div className="metric" key={x[0]}><small>{x[0]}</small><strong>{x[1]}</strong></div>)}</div></div>
    <div className={digitizer?.production_ready||digitizer?.fallback_enabled?"engineReady":"engineWarning"}><b>{digitizer?.active_engine==="classical_fallback"?"Responsive paper ECG digitization is ready":"ECGLight image digitization is ready"}</b><p>{digitizer?.active_engine==="classical_fallback"?"The built-in digitizer extracts a waveform trace and passes it to the classifier. ECGLight can be selected with SMART_ECG_PREFER_CLASSICAL_FALLBACK=0.":"Paper ECGs are reconstructed by ECGLight and Lead II is analyzed by the existing SmartECG classifier."}</p></div><Disclaimer/>
  </>
}

function Prediction({file,setFile,analyze,analysis,busy,digitizer}){
  const [downloadBusy,setDownloadBusy]=useState(false);
  const chartData=analysis?Object.entries(analysis.prediction_distribution).map(([c,v])=>({c,v})):[];
  const isImage=file && /\.(png|jpe?g|pdf)$/i.test(file.name);

  async function downloadWFDB(){
    if(!file || !isImage) return;
    setDownloadBusy(true);
    try{
      const form=new FormData();form.append("file",file);
      const res=await axios.post(`${API}/digitize-image`,form,{responseType:"blob"});
      const url=URL.createObjectURL(res.data);
      const a=document.createElement("a");a.href=url;a.download="smart_ecg_digitized_wfdb.zip";a.click();
      URL.revokeObjectURL(url);
    }catch(e){
      alert("Could not create the WFDB conversion: "+(e.response?.data?.detail||e.message));
    }finally{setDownloadBusy(false);}
  }

  return <>
    <div className="head"><h1>ECG Prediction</h1><p>{digitizer?.production_ready||digitizer?.fallback_enabled?"Upload a paper ECG photo/scan or a digital WFDB record.":"WFDB ZIP analysis is available. Paper-image analysis is unavailable until an image digitizer is configured."} The same trained classifier is used after preprocessing.</p></div>
    <div className="uploadCard">
      <div className="uploadIcon"><Upload/></div>
      <h2>Upload ECG</h2>
      <p><b>Paper ECG:</b> JPG, PNG or PDF input is {digitizer?.active_engine==="classical_fallback"?"available through the built-in image digitizer.":digitizer?.production_ready?"available through ECGLight.":"currently unavailable because no image digitizer is configured."}</p>
      <p><b>For research:</b> upload a WFDB ZIP containing matching <b>.hea</b>, <b>.dat</b> and <b>.atr</b> files.</p>
      <label className="fileButton"><input type="file" accept={ACCEPTED} onChange={e=>setFile(e.target.files?.[0]||null)}/>{file?file.name:"Choose ECG file"}</label>
      {file && <div className="fileHint">{isImage ? digitizer?.active_engine==="classical_fallback" ? "The built-in digitizer will extract the waveform before ML analysis." : digitizer?.production_ready ? "ECGLight will reconstruct the waveform before ML analysis." : "Unavailable: configure an image digitizer." : "WFDB mode: the existing annotated-record pipeline will be used."}</div>}
      <button className="primary" disabled={busy||!file||(isImage&&!digitizer?.production_ready&&!digitizer?.fallback_enabled)} onClick={analyze}>{busy?"Analyzing ECG…":"Analyze ECG"}</button>
    </div>

    {analysis?.conversion && <div className="card conversion">
      <h3>Paper ECG → Digital ECG</h3>
      <div className="conversionGrid">
        <div><small>Input</small><b>{analysis.conversion.source_type?.toUpperCase()}</b></div>
        <div><small>Detected beats</small><b>{analysis.conversion.detected_beats}</b></div>
        <div><small>Sampling rate</small><b>{analysis.conversion.sampling_frequency} Hz</b></div>
        <div><small>Generated files</small><b>.hea + .dat + .atr</b></div>
      </div>
      <p className="muted">Digitization engine: <b>{analysis.conversion.engine}</b>. {analysis.conversion.lead_note}</p><p className="muted">The generated annotation file marks automatically detected QRS locations for beat segmentation only; it does not contain clinician diagnoses.</p>
      <button className="secondary" disabled={downloadBusy} onClick={downloadWFDB}>{downloadBusy?"Preparing...":"Download generated WFDB ZIP"}</button>
    </div>}

    {analysis&&<>
      <div className="grid">
        <div className="card"><small>Input Mode</small><strong>{analysis.input_mode}</strong></div>
        <div className="card"><small>Records Found</small><strong>{analysis.records_found}</strong></div>
        <div className="card"><small>Beats Analyzed</small><strong>{analysis.beats_analyzed}</strong></div>
        <div className="card"><small>Beats Skipped</small><strong>{analysis.beats_skipped}</strong></div>
      </div>
      <div className={`overallAssessment ${analysis.overall_assessment?.status || "inconclusive"}`}>
        <small>Overall model result — research use only</small>
        <h2>{analysis.overall_assessment?.label || "Overall result unavailable"}</h2>
        <p>{analysis.overall_assessment?.explanation || "The server did not return an overall model summary."}</p>
        <div className="assessmentCounts">
          <span>Normal beats: <b>{analysis.overall_assessment?.normal_beats ?? analysis.prediction_distribution.N ?? 0}</b></span>
          <span>Abnormal-class beats: <b>{analysis.overall_assessment?.abnormal_beats ?? ["S","V","F"].reduce((sum, c) => sum + (analysis.prediction_distribution[c] || 0), 0)}</b></span>
          <span>Unknown/other beats: <b>{analysis.overall_assessment?.inconclusive_beats ?? analysis.prediction_distribution.Q ?? 0}</b></span>
        </div>
      </div>
      <div className="card chart"><h3>Prediction Distribution</h3><ResponsiveContainer width="100%" height="100%"><BarChart data={chartData}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="c"/><YAxis/><Tooltip/><Bar dataKey="v" name="Beats"/></BarChart></ResponsiveContainer></div>
      <div className="card"><h3>First Analyzed Beat</h3><table><tbody>
        <tr><th>Record</th><td>{analysis.first_result.record}</td></tr>
        <tr><th>Sample</th><td>{analysis.first_result.sample}</td></tr>
        <tr><th>Detected marker</th><td>{analysis.first_result.annotation}</td></tr>
        <tr><th>Prediction</th><td><b>{analysis.first_result.prediction}</b> — {analysis.first_result.description}</td></tr>
        <tr><th>Probability</th><td>{(analysis.first_result.probability*100).toFixed(2)}%</td></tr>
      </tbody></table></div>
      <div className="card"><h3>Class Probabilities — First Beat</h3><div className="chartSmall"><ResponsiveContainer width="100%" height="100%"><BarChart data={Object.entries(analysis.first_result.probabilities).map(([c,p])=>({c,p:p*100}))}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="c"/><YAxis domain={[0,100]}/><Tooltip/><Bar dataKey="p" name="Probability"/></BarChart></ResponsiveContainer></div></div>
    </>}
    <Disclaimer/>
  </>
}

function Analysis({features}){
  return <><div className="head"><h1>Feature Analysis</h1><p>Exact 23 features used by the final model.</p></div><div className="card featurelist">{features.map((f,i)=><div key={f}><span>{i+1}</span><b>{f}</b></div>)}</div></>
}

function Performance({data}){
  const rows=data?.classes||{};
  return <><div className="head"><h1>Model Performance</h1><p>Nested patient-wise validation.</p></div><div className="grid">{[["Accuracy","86.37%"],["Balanced Accuracy","59.72%"],["Macro F1","54.73%"],["Weighted F1","87.63%"]].map(x=><div className="card" key={x[0]}><small>{x[0]}</small><strong>{x[1]}</strong></div>)}</div><div className="card"><h3>Per-Class Metrics</h3><table><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead><tbody>{Object.entries(rows).map(([c,v])=><tr key={c}><td><b>{c}</b> — {names[c]}</td><td>{(v.precision*100).toFixed(2)}%</td><td>{(v.recall*100).toFixed(2)}%</td><td>{(v.f1*100).toFixed(2)}%</td><td>{v.support}</td></tr>)}</tbody></table></div></>
}

function About(){return <><div className="head"><h1>About SmartECG-ML</h1><p>ECG image digitization + beat-level machine learning research platform.</p></div><div className="card"><h2>SmartECG-ML</h2><p>When ECGLight's runtime and official model weights are available, paper ECG waveforms can be passed through the existing HistGradientBoostingClassifier using the final 23-feature representation.</p><h3>Supported classes</h3><p>N — Normal · S — Supraventricular · V — Ventricular · F — Fusion · Q — Unknown/Other</p><h3>Inputs</h3><p>WFDB ZIP analysis is available. Paper ECG JPG, PNG and PDF analysis requires ECGLight's source, runtime imports, and all four official digitization weights.</p></div><Disclaimer/></>}
function Disclaimer(){return <div className="notice"><Info size={18}/><div><b>Research / Educational Use Only</b><p>This system has not been clinically validated and must not be used for medical diagnosis or treatment decisions.</p></div></div>}

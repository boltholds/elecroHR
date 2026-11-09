const $ = (id) => document.getElementById(id);
let ws, context, microphone, captureNode, sourceNode, sessionId;
let muted = false, generation = -1, audioFinished = false, scheduled = 0, playing = 0;
let sources = [], textLines = [];

function addLine(role, text) {
  if (!text.trim()) return;
  if ($("transcript").querySelector(".empty")) $("transcript").innerHTML = "";
  const entry = document.createElement("div");
  entry.className = "message " + (role === "candidate" ? "user" : "agent");
  const title = document.createElement("span");
  title.className = "role";
  title.textContent = role === "candidate" ? "Кандидат" : "AI-рекрутер";
  entry.appendChild(title);
  entry.append(document.createTextNode(text));
  $("transcript").appendChild(entry);
  $("transcript").scrollTop = $("transcript").scrollHeight;
  textLines.push({role, text});
}

function stopPlayback() {
  for (const src of sources) {
    src.onended = null;
    try { src.stop(); } catch (_) {}
    try { src.disconnect(); } catch (_) {}
  }
  sources = [];
  playing = 0;
  scheduled = context ? context.currentTime : 0;
  audioFinished = false;
}
function acknowledgeIfDone() {
  if (audioFinished && playing === 0 && ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({type: "playback_done", generation}));
    audioFinished = false;
  }
}
function playPCM(bytes) {
  if (!context || generation < 0) return;
  const data = new DataView(bytes);
  const count = Math.floor(data.byteLength / 2);
  const buffer = context.createBuffer(1, count, 24000);
  const channel = buffer.getChannelData(0);
  for (let i = 0; i < count; i++) channel[i] = data.getInt16(i * 2, true) / 32768;
  const src = context.createBufferSource();
  src.buffer = buffer;
  src.connect(context.destination);
  sources.push(src);
  playing++;
  src.onended = () => {
    playing--;
    sources = sources.filter(x => x !== src);
    src.disconnect();
    acknowledgeIfDone();
  };
  scheduled = Math.max(scheduled, context.currentTime + 0.025);
  src.start(scheduled);
  scheduled += buffer.duration;
}
async function releaseAudio() {
  stopPlayback();
  try { captureNode?.disconnect(); sourceNode?.disconnect(); } catch (_) {}
  microphone?.getTracks().forEach(t => t.stop());
  if (context && context.state !== "closed") await context.close();
  context = undefined;
}
async function start() {
  if (!$("consent").checked) {
    alert("Для тестового интервью нужно согласие на сохранение текста.");
    return;
  }
  $("start").disabled = true;
  $("status").textContent = "Подключение...";
  try {
    // Get microphone permission before reserving the single backend session.
    microphone = await navigator.mediaDevices.getUserMedia({
      audio: {echoCancellation: true, noiseSuppression: true, autoGainControl: true},
    });
    context = new AudioContext();
    await context.audioWorklet.addModule("/static/audio-worklet.js");
    captureNode = new AudioWorkletNode(context, "pcm16-encoder");
    sourceNode = context.createMediaStreamSource(microphone);
    sourceNode.connect(captureNode);
    captureNode.connect(context.destination);
    await context.resume();
    const res = await fetch("/sessions", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({consent: true}),
    });
    if (!res.ok) throw new Error("Невозможно начать интервью: HTTP " + res.status);
    sessionId = (await res.json()).session_id;
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(protocol + "//" + location.host + "/ws/" + sessionId);
    ws.binaryType = "arraybuffer";
    captureNode.port.onmessage = ({data}) => {
      if (!muted && ws?.readyState === WebSocket.OPEN) ws.send(data);
    };
    ws.onmessage = ({data}) => {
      if (data instanceof ArrayBuffer) { playPCM(data); return; }
      const event = JSON.parse(data);
      switch (event.type) {
        case "state":
          $("status").textContent = {
            created: "Подключение", listening: "Слушаю кандидата",
            thinking: "Распознавание и ответ", speaking: "Говорит AI",
            completed: "Интервью завершено",
          }[event.state] || event.state;
          $("pulse").classList.toggle("active", event.state === "speaking");
          break;
        case "audio_start":
          stopPlayback();
          generation = event.generation;
          break;
        case "audio_end":
          if (event.generation === generation) {
            audioFinished = true; acknowledgeIfDone();
          }
          break;
        case "agent_text": addLine("agent", event.text); break;
        case "candidate_text": addLine("candidate", event.text); break;
        case "interrupt": stopPlayback(); break;
        case "rag":
          $("rag").textContent = "RAG: " + event.sources.map(s => s.source).join(", ");
          break;
        case "error":
          $("status").textContent = "Ошибка: " + event.code;
          break;
      }
    };
    ws.onopen = () => { $("end").disabled = false; $("mute").disabled = false; };
    ws.onclose = () => {
      if (!$("end").disabled) $("status").textContent = "Соединение разорвано";
      $("end").disabled = true;
      $("mute").disabled = true;
    };
  } catch (err) {
    $("status").textContent = String(err.message || err);
    $("start").disabled = false;
    await releaseAudio();
  }
}
async function finish() {
  $("end").disabled = true;
  $("status").textContent = "Формирование отчёта...";
  ws?.close();
  await releaseAudio();
  if (!sessionId) return;
  try {
    const res = await fetch("/sessions/" + sessionId + "/finish", {method: "POST"});
    if (!res.ok) throw new Error("Ошибка формирования отчёта: HTTP " + res.status);
    const report = await res.json();
    $("report").textContent = JSON.stringify(report, null, 2);
    $("result").hidden = false;
    $("status").textContent = "Завершено: " + report.score + "/100 (проверяет HR)";
  } catch (err) {
    $("status").textContent = String(err.message || err);
  } finally {
    $("start").disabled = false;
  }
}
$("start").addEventListener("click", start);
$("end").addEventListener("click", finish);
$("mute").addEventListener("click", () => {
  muted = !muted;
  $("mute").textContent = muted ? "Включить микрофон" : "Выключить микрофон";
});

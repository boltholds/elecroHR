const $ = (id) => document.getElementById(id);
let ws, context, microphone, captureNode, sourceNode, sessionId;
let muted = false, generation = -1, audioFinished = false, scheduled = 0, playing = 0;
let sources = [], textLines = [];
let playbackStartedAt = 0;
let timingHistory = new Map();
let timingSummary = {completed_turns: 0, average_ms: {}, p95_ms: {}};

const TIMING_LABELS = [
  ["vad_cpu_ms", "VAD — CPU"],
  ["speech_capture_ms", "Запись фразы"],
  ["stt_ms", "Whisper STT"],
  ["react_llm_ms", "ReAct — выбор действия"],
  ["vacancy_rag_ms", "RAG вакансии"],
  ["question_rag_ms", "RAG вопросов"],
  ["dialogue_ms", "Диалог всего"],
  ["llm_http_ms", "LLM — ответ (HTTP)"],
  ["ollama_load_ms", "Ollama — загрузка"],
  ["ollama_prompt_ms", "Ollama — prompt evaluation"],
  ["ollama_decode_ms", "Ollama — генерация токенов"],
  ["tts_http_ms", "TTS — полный запрос"],
  ["tts_model_ms", "XTTS — инференс"],
  ["websocket_send_ms", "WebSocket — отправка"],
  ["first_audio_ms", "До первого аудио"],
  ["backend_ms", "Backend до audio_end"],
  ["client_playback_ms", "Воспроизведение браузером"],
  ["playback_ack_ms", "Ожидание подтверждения"],
  ["end_to_end_ms", "До конца воспроизведения"],
];

function formatMs(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return value >= 1000 ? (value / 1000).toFixed(2) + " с" : value.toFixed(1) + " мс";
}
const OLLAMA_OPERATIONS = {
  availability: "Знакомство: удобное время",
  name: "Знакомство: имя",
  address: "Знакомство: обращение",
  confirmation: "Подтверждение",
  react: "ReAct · маршрутизация",
  answer: "LLM · ответ кандидату",
  scoring: "Оценка интервью",
};
function formatRate(value) {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(1) + " ток/с" : "—";
}
function renderOllama(calls, summary) {
  const list = $("ollama-calls");
  list.replaceChildren();
  if (!calls?.length) {
    const empty = document.createElement("p");
    empty.className = "timing-note";
    empty.textContent = "В этой реплике Ollama не вызывалась.";
    list.appendChild(empty);
  } else {
    for (const call of calls) {
      const item = document.createElement("div");
      item.className = "ollama-call";
      const heading = document.createElement("strong");
      heading.textContent = (OLLAMA_OPERATIONS[call.operation] || call.operation)
        + (call.status === "error" ? " · ошибка" : "");
      item.appendChild(heading);
      const details = document.createElement("div");
      details.className = "ollama-call-details";
      const values = [
        ["HTTP", formatMs(call.wall_ms)],
        ["В Ollama", formatMs(call.total_ms)],
        ["Загрузка", formatMs(call.load_ms)],
        ["Prompt eval", formatMs(call.prompt_ms)],
        ["Генерация", formatMs(call.decode_ms)],
        ["Входных токенов", call.input_tokens ?? "—"],
        ["Выходных токенов", call.output_tokens ?? "—"],
        ["Скорость генерации", formatRate(call.decode_tokens_per_s)],
        ["Скорость промпта", formatRate(call.prompt_tokens_per_s)],
      ];
      for (const [name, value] of values) {
        const cell = document.createElement("div");
        const label = document.createElement("span");
        label.textContent = name;
        const number = document.createElement("b");
        number.textContent = String(value);
        cell.append(label, number);
        details.appendChild(cell);
      }
      item.appendChild(details);
      list.appendChild(item);
    }
  }
  const operations = summary?.ollama_summary || {};
  const summaryLines = Object.entries(operations).map(([name, stats]) =>
    (OLLAMA_OPERATIONS[name] || name) +
    ": " + stats.calls + " выз. · HTTP ср. " + formatMs(stats.avg_wall_ms) +
    " / p95 " + formatMs(stats.p95_wall_ms) +
    " · генерация " + formatRate(stats.avg_decode_tokens_per_s)
  );
  $("ollama-summary").textContent = summaryLines.length
    ? summaryLines.join("\n") : "Сводка появится после завершения реплик.";
}

function showTiming(event) {
  timingHistory.set(event.timing.turn_id, event.timing);
  timingSummary = event.summary || timingSummary;
  const last = event.timing;
  renderOllama(last.ollama_calls, timingSummary);
  const stateNames = {
    running: "обрабатывается", completed: "завершено",
    interrupted: "прервано", failed: "ошибка", no_speech: "нет речи",
  };
  $("timing-turn").textContent =
    "Реплика #" + last.turn_id + " · " + (last.kind === "greeting" ? "приветствие" : last.phase) +
    " · " + (stateNames[last.status] || last.status);

  const grid = $("timing-grid");
  grid.replaceChildren();
  for (const [key, label] of TIMING_LABELS) {
    const cell = document.createElement("div");
    cell.className = "timing-item";
    const name = document.createElement("span");
    name.textContent = label;
    const value = document.createElement("strong");
    value.textContent = formatMs(last.stages_ms[key]);
    cell.append(name, value);
    grid.appendChild(cell);
  }
  $("timing-summary").textContent =
    "Завершённых реплик: " + timingSummary.completed_turns +
    " · среднее до первого аудио: " + formatMs(timingSummary.average_ms?.first_audio_ms) +
    " · p95: " + formatMs(timingSummary.p95_ms?.first_audio_ms) +
    " · среднее полное время: " + formatMs(timingSummary.average_ms?.end_to_end_ms);
  const rows = Array.from(timingHistory.values()).slice(-12).reverse();
  $("timing-history").textContent = rows.map(item => (
    "#" + item.turn_id + " (" + item.status + ") — STT " +
    formatMs(item.stages_ms.stt_ms) + ", LLM " +
    formatMs(item.stages_ms.llm_http_ms || item.stages_ms.react_llm_ms) +
    ", TTS " + formatMs(item.stages_ms.tts_http_ms) +
    ", всего " + formatMs(item.stages_ms.end_to_end_ms || item.stages_ms.backend_ms)
  )).join("\n");
}


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
  playbackStartedAt = 0;
}
function acknowledgeIfDone() {
  if (audioFinished && playing === 0 && ws?.readyState === WebSocket.OPEN) {
    const playbackMs = playbackStartedAt > 0 ? performance.now() - playbackStartedAt : 0;
    ws.send(JSON.stringify({type: "playback_done", generation, playback_ms: playbackMs}));
    audioFinished = false;
    playbackStartedAt = 0;
  }
}
function playPCM(bytes) {
  if (!context || generation < 0) return;
  if (playbackStartedAt === 0) playbackStartedAt = performance.now();
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
  timingHistory = new Map();
  timingSummary = {completed_turns: 0, average_ms: {}, p95_ms: {}};
  $("timing-grid").replaceChildren();
  renderOllama([], timingSummary);
  $("timing-summary").textContent = "Измерения начнутся после подключения.";
  $("timing-history").textContent = "";
  $("timing-turn").textContent = "Нет данных";
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
        case "timing": showTiming(event); break;
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
        case "dialogue": {
          const p = event.profile;
          const stages = {
            availability: "Уточняем удобное время",
            name: "Знакомство: имя",
            address: "Знакомство: обращение",
            confirmation: "Подтверждение обращения",
            screening: "Техническое интервью",
            finished: "Интервью завершено",
          };
          $("dialogue-stage").textContent = "Этап: " + (stages[p.phase] || p.phase);
          $("candidate-profile").textContent = p.confirmed
            ? p.name + " · на «" + p.form + "»"
            : "Имя и обращение пока не подтверждены";
          break;
        }
        case "session_finished": void finish(); break;
        case "question_bank":
          $("question-bank").textContent = "Банк вопросов: " + event.total + " вопросов";
          break;
        case "question":
          $("rag").textContent = "Вопрос из RAG: " + event.question_id;
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
    try {
      const metricsResponse = await fetch("/sessions/" + sessionId + "/metrics");
      if (metricsResponse.ok) {
        const metrics = await metricsResponse.json();
        $("timing-history").textContent += "\n\nИтог по сессии:\n" + JSON.stringify(metrics, null, 2);
      }
    } catch (_) { /* Nonessential diagnostics. */ }
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

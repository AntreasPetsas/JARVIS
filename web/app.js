(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const els = {
    conn: $("conn"), clock: $("clock"), date: $("date"),
    stateLabel: $("state-label"), subtitle: $("subtitle"), name: $("assistant-name"),
    weather: $("weather-body"), software: $("software-body"),
    hobby: $("hobby-body"), reminders: $("reminders-body"),
    nowplaying: $("nowplaying-body"),
    transcript: $("transcript"), form: $("cmd-form"), input: $("cmd-input"),
    brief: $("brief-btn"), mic: $("mic-btn"),
  };

  // ---------- Clock ----------
  const tick = () => {
    const d = new Date();
    els.clock.textContent = d.toLocaleTimeString("en-GB");
    els.date.textContent = d.toLocaleDateString("en-GB", { weekday: "short", day: "2-digit", month: "short" });
  };
  setInterval(tick, 1000); tick();

  // ---------- Reactor canvas ----------
  const canvas = $("reactor"), ctx = canvas.getContext("2d");
  const SIZE = 440, dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = SIZE * dpr; canvas.height = SIZE * dpr; ctx.scale(dpr, dpr);

  const STYLE = {
    offline:   { color: "#ff6b6b", amp: 0.02, label: "OFFLINE" },
    idle:      { color: "#36e6ff", amp: 0.10, label: "STANDBY" },
    listening: { color: "#46ffb0", amp: 0.30, label: "LISTENING" },
    thinking:  { color: "#ffb347", amp: 0.16, label: "PROCESSING" },
    speaking:  { color: "#7fefff", amp: 0.72, label: "SPEAKING" },
  };
  let state = "offline", amp = 0.02, voiceLevel = null;
  const BARS = 72, seed = Array.from({ length: BARS }, () => Math.random());

  const setState = (s) => {
    if (!STYLE[s]) return;
    state = s; els.stateLabel.textContent = STYLE[s].label;
    els.mic.classList.toggle("live", s === "listening");
  };

  const hexA = (hex, a) => {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  };

  function ring(cx, cy, r, rot, alpha, color, segs) {
    ctx.save(); ctx.translate(cx, cy); ctx.rotate(rot);
    ctx.strokeStyle = hexA(color, alpha); ctx.lineWidth = 2;
    const gap = (Math.PI * 2) / segs;
    for (let i = 0; i < segs; i++) { ctx.beginPath(); ctx.arc(0, 0, r, i * gap, i * gap + gap * 0.62); ctx.stroke(); }
    ctx.restore();
  }
  function ticks(cx, cy, r, rot, color) {
    ctx.save(); ctx.translate(cx, cy); ctx.rotate(rot);
    ctx.strokeStyle = hexA(color, 0.5); ctx.lineWidth = 1.5;
    for (let i = 0; i < 60; i++) {
      const a = (i / 60) * Math.PI * 2, long = i % 5 === 0, e = r + (long ? 10 : 5);
      ctx.beginPath(); ctx.moveTo(Math.cos(a) * r, Math.sin(a) * r); ctx.lineTo(Math.cos(a) * e, Math.sin(a) * e); ctx.stroke();
    }
    ctx.restore();
  }
  function draw(t) {
    const s = STYLE[state] || STYLE.idle, cx = SIZE / 2, cy = SIZE / 2;
    let goal = s.amp;
    if (audioPlaying && analyser) {
      analyser.getByteFrequencyData(freqData);
      let sum = 0; for (let i = 0; i < freqData.length; i++) sum += freqData[i];
      voiceLevel = sum / freqData.length / 255;
      goal = 0.18 + voiceLevel * 0.95;            // reactor pulses to the real voice
    } else if (state === "speaking") {
      goal = s.amp * (0.55 + 0.45 * Math.abs(Math.sin(t / 90)));
    }
    amp += (goal - amp) * 0.08;
    const color = s.color;
    ctx.clearRect(0, 0, SIZE, SIZE);

    const coreR = 56 + amp * 30;
    const g = ctx.createRadialGradient(cx, cy, 4, cx, cy, coreR * 1.8);
    g.addColorStop(0, color); g.addColorStop(0.35, hexA(color, 0.5)); g.addColorStop(1, "transparent");
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(cx, cy, coreR * 1.8, 0, Math.PI * 2); ctx.fill();

    ctx.lineWidth = 2; ctx.strokeStyle = hexA(color, 0.9);
    ctx.beginPath(); ctx.arc(cx, cy, coreR, 0, Math.PI * 2); ctx.stroke();

    const baseR = 110;
    for (let i = 0; i < BARS; i++) {
      const a = (i / BARS) * Math.PI * 2;
      const noise = 0.5 + 0.5 * Math.sin(t / 200 + seed[i] * 10 + i);
      const len = 8 + amp * 70 * noise;
      ctx.strokeStyle = hexA(color, 0.35 + 0.5 * noise); ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(cx + Math.cos(a) * baseR, cy + Math.sin(a) * baseR);
      ctx.lineTo(cx + Math.cos(a) * (baseR + len), cy + Math.sin(a) * (baseR + len));
      ctx.stroke();
    }
    ring(cx, cy, 158, t / 1400, 0.85, color, 14);
    ring(cx, cy, 178, -t / 1000, 0.6, color, 8);
    ticks(cx, cy, 200, t / 2600, color);
    requestAnimationFrame(draw);
  }
  requestAnimationFrame(draw);

  // ---------- Speech (browser TTS — replaced by edge-tts in Layer 2) ----------
  let voices = [];
  const loadVoices = () => { voices = window.speechSynthesis ? speechSynthesis.getVoices() : []; };
  if (window.speechSynthesis) { loadVoices(); speechSynthesis.onvoiceschanged = loadVoices; }
  function pickVoice() {
    const pref = ["Google UK English Male", "Microsoft Ryan", "Microsoft George", "Daniel", "en-GB"];
    for (const p of pref) { const v = voices.find((v) => v.name.includes(p) || v.lang === p); if (v) return v; }
    return voices.find((v) => v.lang && v.lang.startsWith("en")) || null;
  }
  function speak(text) {
    const synth = window.speechSynthesis;
    const ms = Math.max(2500, text.split(/\s+/).length * 380);
    if (!synth) {
      setState("speaking");
      clearTimeout(speak._t); speak._t = setTimeout(() => setState("idle"), ms);
      return;
    }
    synth.cancel();
    const u = new SpeechSynthesisUtterance(text);
    const v = pickVoice(); if (v) u.voice = v;
    u.rate = 1.0; u.pitch = 0.9;
    u.onstart = () => setState("speaking");
    u.onend = () => setState("idle");
    u.onerror = () => setState("idle");
    synth.speak(u);
    clearTimeout(speak._t); speak._t = setTimeout(() => { if (state === "speaking") setState("idle"); }, ms);
  }

  // ---------- High-quality voice (edge-tts) playback + live waveform ----------
  let audioCtx = null, analyser = null, freqData = null, audioPlaying = false, currentSrc = null;
  function stopAudio() {
    if (currentSrc) { try { currentSrc.stop(); } catch (_) {} currentSrc = null; }
    if (synth) synth.cancel();
    audioPlaying = false; voiceLevel = null; setState("idle");
  }
  function ensureAudio() {
    if (!audioCtx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return;
      audioCtx = new AC();
      analyser = audioCtx.createAnalyser();
      analyser.fftSize = 128;
      freqData = new Uint8Array(analyser.frequencyBinCount);
      analyser.connect(audioCtx.destination);
    }
    if (audioCtx.state === "suspended") audioCtx.resume();
  }
  async function playAudioB64(b64) {
    ensureAudio();
    if (!audioCtx) throw new Error("no audio context");
    const bin = atob(b64), bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const buf = await audioCtx.decodeAudioData(bytes.buffer);
    const src = audioCtx.createBufferSource();
    src.buffer = buf; src.connect(analyser);
    currentSrc = src;
    setState("speaking"); audioPlaying = true;
    send({ type: "speaking", on: true, seconds: buf.duration }); // pause wake word so we don't hear ourselves
    let finished = false;
    const finish = () => {
      if (finished) return;
      finished = true;
      if (currentSrc === src) currentSrc = null;
      audioPlaying = false; voiceLevel = null; setState("idle");
      send({ type: "speaking", on: false });
    };
    src.onended = finish;
    src.start();
    setTimeout(finish, buf.duration * 1000 + 800); // failsafe if onended never fires
  }

  // ---------- Renderers ----------
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  function renderWeather(w) {
    if (!w || !w.ok) { els.weather.innerHTML = `<div class="muted">${esc((w && w.error) || "Weather unavailable.")}</div>`; return; }
    els.weather.innerHTML = `
      <div class="wx-main">
        <div class="wx-icon">${w.icon}</div>
        <div>
          <div class="wx-temp">${w.temp}${w.unit}</div>
          <div class="wx-cond">${esc(w.condition)}</div>
          <div class="wx-city">${esc(w.city)}${w.country ? ", " + esc(w.country) : ""}</div>
        </div>
      </div>
      <div class="wx-grid">
        <span class="k">Feels like</span><span class="v">${w.feels_like}${w.unit}</span>
        <span class="k">High / Low</span><span class="v">${w.high}° / ${w.low}°</span>
        <span class="k">Humidity</span><span class="v">${w.humidity ?? "—"}%</span>
        <span class="k">Wind</span><span class="v">${w.wind} ${w.wind_unit}</span>
        <span class="k">Precip.</span><span class="v">${w.precip_chance ?? 0}%</span>
      </div>`;
  }
  function renderFeed(el, items) {
    if (!items || !items.length) { el.innerHTML = `<div class="muted">No items.</div>`; return; }
    el.innerHTML = `<ul class="feed">` + items.map((a) =>
      `<li><a href="${esc(a.url)}" target="_blank" rel="noreferrer">${esc(a.title)}</a><span class="src">${esc(a.source)}</span></li>`
    ).join("") + `</ul>`;
  }
  function renderNowPlaying(np) {
    if (!np || !np.title) {
      els.nowplaying.innerHTML = `<div class="muted">${esc((np && np.message) || "Spotify idle.")}</div>`;
      return;
    }
    const art = np.image
      ? `<img class="np-art" src="${esc(np.image)}" alt="" />`
      : `<div class="np-art np-art--empty">♪</div>`;
    els.nowplaying.innerHTML = `
      <div class="np">
        ${art}
        <div class="np-meta">
          <div class="np-title">${esc(np.title)}</div>
          <div class="np-artist">${esc(np.artist)}</div>
          ${np.album ? `<div class="np-album">${esc(np.album)}</div>` : ""}
          <div class="np-status">${np.is_playing ? "▶ PLAYING" : "❚❚ PAUSED"}</div>
        </div>
      </div>`;
  }
  function renderReminders(items) {
    if (!items || !items.length) { els.reminders.innerHTML = `<div class="muted">No tasks logged.</div>`; return; }
    els.reminders.innerHTML = `<ul class="todo">` + items.map((r) =>
      `<li class="${r.done ? "done" : ""}" data-id="${r.id}" title="Click to mark done"><span class="box"></span><span>${esc(r.text)}</span></li>`
    ).join("") + `</ul>`;
  }
  function addLine(role, text) {
    const div = document.createElement("div");
    div.className = "line " + role;
    div.textContent = (role === "jarvis" ? "J: " : "» ") + text;
    els.transcript.appendChild(div);
    while (els.transcript.children.length > 3) els.transcript.removeChild(els.transcript.firstChild);
    setTimeout(() => div.remove(), 9000);
  }

  // ---------- WebSocket ----------
  let ws, retry = 0;
  function connect() {
    ws = new WebSocket(`ws://${location.host}/ws`);
    ws.onopen = () => {
      retry = 0;
      els.conn.textContent = "ONLINE"; els.conn.className = "badge online";
      setState("idle"); els.subtitle.textContent = "Systems nominal. Awaiting your command.";
    };
    ws.onclose = () => {
      els.conn.textContent = "OFFLINE"; els.conn.className = "badge offline";
      setState("offline"); els.subtitle.textContent = "Connection lost. Reconnecting…";
      retry = Math.min(retry + 1, 6); setTimeout(connect, retry * 800);
    };
    ws.onmessage = (ev) => {
      let m; try { m = JSON.parse(ev.data); } catch { return; }
      switch (m.type) {
        case "hello": els.name.textContent = String(m.name || "JARVIS").toUpperCase().split("").join("."); break;
        case "state": setState(m.state); break;
        case "say":
          els.subtitle.textContent = m.text;
          if (m.audio) playAudioB64(m.audio).catch(() => speak(m.text));
          else speak(m.text);
          break;
        case "subtitle": els.subtitle.textContent = m.text; break;
        case "voice_status":
          if (m.message) els.subtitle.textContent = m.message;
          if (m.ok === false && m.error) console.warn("voice:", m.error);
          break;
        case "stop_audio": stopAudio(); break;
        case "transcript": addLine(m.role, m.text); break;
        case "panel":
          if (m.panel === "weather") renderWeather(m.data);
          else if (m.panel === "software_news") renderFeed(els.software, m.data);
          else if (m.panel === "hobby_news") renderFeed(els.hobby, m.data);
          else if (m.panel === "reminders") renderReminders(m.data);
          else if (m.panel === "nowplaying") renderNowPlaying(m.data);
          break;
      }
    };
  }
  const send = (obj) => { if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj)); };
  connect();

  // ---------- Input ----------
  els.form.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = els.input.value.trim();
    if (!text) return;
    send({ type: "command", text });
    els.input.value = "";
  });
  els.brief.addEventListener("click", () => send({ type: "briefing" }));
  els.mic.addEventListener("click", () => {
    ensureAudio();
    send({ type: "listen" });
    setState("listening");
    els.subtitle.textContent = "Listening… speak now.";
  });
  // Click a reminder to mark it done.
  els.reminders.addEventListener("click", (e) => {
    const li = e.target.closest("li[data-id]");
    if (li) send({ type: "reminder_done", id: Number(li.dataset.id) });
  });

  // Browsers block audio until the user interacts — unlock the AudioContext once.
  const unlock = () => {
    ensureAudio();
    window.removeEventListener("pointerdown", unlock);
    window.removeEventListener("keydown", unlock);
  };
  window.addEventListener("pointerdown", unlock);
  window.addEventListener("keydown", unlock);
})();

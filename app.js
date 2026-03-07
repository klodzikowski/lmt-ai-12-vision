const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const statusBadge = document.getElementById("statusBadge");
const objectList = document.getElementById("objectList");
const emotionList = document.getElementById("emotionList");
const startButton = document.getElementById("startButton");
const stopButton = document.getElementById("stopButton");
const thresholdRange = document.getElementById("thresholdRange");
const thresholdValue = document.getElementById("thresholdValue");
const emptyState = document.getElementById("emptyState");
const objectModelOutput = document.getElementById("objectModelOutput");
const faceDetectorOutput = document.getElementById("faceDetectorOutput");
const expressionModelOutput = document.getElementById("expressionModelOutput");
const runtimeModelOutput = document.getElementById("runtimeModelOutput");

const ctx = overlay.getContext("2d");
const FACE_MODEL_URLS = [
  "https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.15/model",
  "https://justadudewhohacks.github.io/face-api.js/models",
];
const OBJECT_MODEL_BASE = "lite_mobilenet_v2";
const INFERENCE_TICK_INTERVAL_MS = 85;
const FACE_DETECT_INTERVAL_MS = 240;
const OBJECT_INFERENCE_TIMEOUT_MS = 5000;
const FACE_INFERENCE_TIMEOUT_MS = 7000;
const MAX_OBJECTS_IN_LIST = 6;
const MAX_FACES_IN_LIST = 4;
const FACE_DETECT_OPTIONS = {
  inputSize: 320,
  scoreThreshold: 0.15,
};

const state = {
  objectModel: null,
  emotionReady: false,
  stream: null,
  animationId: null,
  isRunning: false,
  inferenceBusy: false,
  latestObjects: [],
  latestFaces: [],
  lastInferenceAt: 0,
  lastFaceInferAt: 0,
  objectLatencyMs: 0,
  faceLatencyMs: 0,
  fps: 0,
  lastFrameAt: 0,
  lastRuntimeUpdateAt: 0,
  objectErrors: 0,
  emotionFailures: 0,
  warnedEmotionFailure: false,
  runEpoch: 0,
  emotionModelSource: "",
};

function setStatus(text, level = "info") {
  statusBadge.textContent = text;
  statusBadge.dataset.level = level;
}

function setModelOutput(el, text, level = "info") {
  if (!el) return;
  el.textContent = text;
  el.dataset.level = level;
}

function objectColor(name) {
  let hash = 0;
  for (const char of name) {
    hash = (hash << 5) - hash + char.charCodeAt(0);
    hash |= 0;
  }
  const hue = Math.abs(hash) % 360;
  return `hsl(${hue} 85% 54%)`;
}

function resizeOverlayToVideo() {
  if (!video.videoWidth || !video.videoHeight) return;
  if (overlay.width === video.videoWidth && overlay.height === video.videoHeight) {
    return;
  }

  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;
}

function drawLabeledBox(x, y, width, height, label, color, lineWidth = 3) {
  const drawX = overlay.width - x - width;

  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.strokeRect(drawX, y, width, height);

  ctx.font = '600 16px "Sora", sans-serif';
  const textWidth = ctx.measureText(label).width;
  const boxHeight = 24;
  const labelY = Math.max(0, y - boxHeight);

  ctx.fillStyle = color;
  ctx.fillRect(drawX, labelY, textWidth + 12, boxHeight);
  ctx.fillStyle = "#09101a";
  ctx.fillText(label, drawX + 6, labelY + 4);
}

function drawDetections(objects, faces) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  ctx.textBaseline = "top";
  ctx.setLineDash([]);

  for (const pred of objects) {
    const [x, y, width, height] = pred.bbox;
    const color = objectColor(pred.class);
    const label = `${pred.class} ${(pred.score * 100).toFixed(0)}%`;
    drawLabeledBox(x, y, width, height, label, color, 3);
  }

  for (const face of faces) {
    const [x, y, width, height] = face.bbox;
    const faceLabel = `${face.emotion} ${(face.emotionScore * 100).toFixed(0)}%`;
    ctx.setLineDash([6, 4]);
    drawLabeledBox(x, y, width, height, faceLabel, "#1fffc6", 2);
    ctx.setLineDash([]);
  }
}

function updateObjectList(objects) {
  const top = objects.slice(0, MAX_OBJECTS_IN_LIST);
  objectList.innerHTML = "";

  if (!top.length) {
    const li = document.createElement("li");
    li.textContent = "No objects above confidence threshold.";
    objectList.appendChild(li);
    return;
  }

  for (const pred of top) {
    const li = document.createElement("li");
    li.textContent = `${pred.class} (${(pred.score * 100).toFixed(0)}%)`;
    objectList.appendChild(li);
  }
}

function updateEmotionList(faces) {
  const top = faces.slice(0, MAX_FACES_IN_LIST);
  emotionList.innerHTML = "";

  if (!top.length) {
    const li = document.createElement("li");
    li.textContent = "No face emotions detected.";
    emotionList.appendChild(li);
    return;
  }

  for (const face of top) {
    const li = document.createElement("li");
    li.textContent =
      `${face.emotion} (${(face.emotionScore * 100).toFixed(0)}%)` +
      ` | face ${(face.faceScore * 100).toFixed(0)}%`;
    emotionList.appendChild(li);
  }
}

function getTopExpression(expressions) {
  const entries = Object.entries(expressions || {});
  if (!entries.length) return { label: "neutral", score: 0 };

  let best = entries[0];
  for (const current of entries) {
    if (current[1] > best[1]) best = current;
  }
  return { label: best[0], score: best[1] };
}

function withTimeout(promise, timeoutMs, label) {
  let timer = null;
  const timeoutPromise = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(label)), timeoutMs);
  });

  return Promise.race([promise, timeoutPromise]).finally(() => {
    if (timer) clearTimeout(timer);
  });
}

function updateRuntimeOutput() {
  const backend =
    window.tf && typeof tf.getBackend === "function" ? tf.getBackend() : "n/a";
  const fpsText = state.fps ? `${state.fps.toFixed(1)} fps` : "warming up";
  const objectText = state.objectLatencyMs
    ? `${state.objectLatencyMs.toFixed(0)}ms obj`
    : "obj -";
  const faceText = state.emotionReady
    ? state.faceLatencyMs
      ? `${state.faceLatencyMs.toFixed(0)}ms face`
      : "face -"
    : "face off";
  const level = backend === "cpu" ? "warn" : "ok";

  setModelOutput(
    runtimeModelOutput,
    `${backend} | ${fpsText} | ${objectText} | ${faceText}`,
    level
  );
}

async function configureTfBackend() {
  if (!window.tf) return;
  await tf.ready();

  if (tf.getBackend() === "webgl") return;

  try {
    await tf.setBackend("webgl");
    await tf.ready();
  } catch (error) {
    console.warn("WebGL backend unavailable, keeping default backend.", error);
  }
}

async function loadEmotionModels() {
  if (typeof faceapi === "undefined") return false;

  for (const source of FACE_MODEL_URLS) {
    try {
      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri(source),
        faceapi.nets.faceExpressionNet.loadFromUri(source),
      ]);
      state.emotionModelSource = source;
      return true;
    } catch (error) {
      console.warn(`Emotion model source failed: ${source}`, error);
    }
  }

  return false;
}

async function loadModels() {
  setStatus("Loading object and emotion models...", "info");
  startButton.disabled = true;
  setModelOutput(objectModelOutput, "Loading...", "info");
  setModelOutput(faceDetectorOutput, "Loading...", "info");
  setModelOutput(expressionModelOutput, "Loading...", "info");
  setModelOutput(runtimeModelOutput, "Initializing...", "info");

  try {
    await configureTfBackend();
    state.objectModel = await cocoSsd.load({ base: OBJECT_MODEL_BASE });
    setModelOutput(objectModelOutput, "Loaded", "ok");
  } catch (error) {
    console.error(error);
    setStatus("Object model load failed. Check internet connection.", "error");
    setModelOutput(objectModelOutput, "Load failed", "error");
    setModelOutput(faceDetectorOutput, "Blocked (no object model)", "warn");
    setModelOutput(expressionModelOutput, "Blocked (no object model)", "warn");
    setModelOutput(runtimeModelOutput, "Runtime unavailable", "error");
    return;
  }

  try {
    state.emotionReady = await loadEmotionModels();
  } catch (error) {
    console.error(error);
    state.emotionReady = false;
  }

  if (state.emotionReady) {
    const sourceName = state.emotionModelSource.includes("vladmandic")
      ? "vladmandic"
      : "face-api.js";
    setModelOutput(faceDetectorOutput, `Loaded (${sourceName})`, "ok");
    setModelOutput(expressionModelOutput, `Loaded (${sourceName})`, "ok");
    setStatus("Object + emotion models loaded. Click Start camera.", "ok");
  } else {
    setModelOutput(faceDetectorOutput, "Unavailable", "warn");
    setModelOutput(expressionModelOutput, "Unavailable", "warn");
    setStatus(
      "Object model loaded. Emotion mode unavailable (object detection still works).",
      "warn"
    );
  }

  startButton.disabled = false;
  updateRuntimeOutput();
}

async function startCamera() {
  if (state.isRunning) return;
  if (!state.objectModel) await loadModels();
  if (!state.objectModel) return;

  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 960 },
        height: { ideal: 540 },
      },
      audio: false,
    });

    video.srcObject = state.stream;
    await video.play();
    resizeOverlayToVideo();

    state.isRunning = true;
    state.runEpoch += 1;
    state.inferenceBusy = false;
    state.lastInferenceAt = 0;
    state.lastFaceInferAt = 0;
    state.objectLatencyMs = 0;
    state.faceLatencyMs = 0;
    state.fps = 0;
    state.lastFrameAt = 0;
    state.lastRuntimeUpdateAt = 0;
    state.objectErrors = 0;
    state.emotionFailures = 0;
    state.warnedEmotionFailure = false;
    state.emotionModelSource = state.emotionModelSource || "";
    state.latestObjects = [];
    state.latestFaces = [];

    updateObjectList([]);
    updateEmotionList([]);
    emptyState.hidden = true;
    startButton.disabled = true;
    stopButton.disabled = false;

    setStatus(
      state.emotionReady
        ? "Detection running (objects + emotions)..."
        : "Detection running (objects only)...",
      "ok"
    );
    setModelOutput(objectModelOutput, "Running...", "ok");
    if (state.emotionReady) {
      setModelOutput(faceDetectorOutput, "Running (waiting for face)...", "ok");
      setModelOutput(expressionModelOutput, "Running (waiting for face)...", "ok");
    }

    state.animationId = requestAnimationFrame(detectLoop);
  } catch (error) {
    console.error(error);
    setStatus("Camera access denied or unavailable.", "error");
  }
}

function stopCamera() {
  state.isRunning = false;
  state.runEpoch += 1;
  state.inferenceBusy = false;
  state.lastInferenceAt = 0;
  state.lastFaceInferAt = 0;
  state.latestObjects = [];
  state.latestFaces = [];

  if (state.animationId) {
    cancelAnimationFrame(state.animationId);
    state.animationId = null;
  }

  if (state.stream) {
    for (const track of state.stream.getTracks()) track.stop();
    state.stream = null;
  }

  video.pause();
  video.srcObject = null;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  updateObjectList([]);
  updateEmotionList([]);
  emptyState.hidden = false;

  startButton.disabled = !state.objectModel;
  stopButton.disabled = true;
  setStatus("Camera stopped.", "info");
  setModelOutput(
    objectModelOutput,
    state.objectModel ? "Loaded (camera off)" : "Waiting...",
    state.objectModel ? "ok" : "info"
  );
  setModelOutput(
    faceDetectorOutput,
    state.emotionReady ? "Loaded (camera off)" : "Unavailable",
    state.emotionReady ? "ok" : "warn"
  );
  setModelOutput(
    expressionModelOutput,
    state.emotionReady ? "Loaded (camera off)" : "Unavailable",
    state.emotionReady ? "ok" : "warn"
  );
  updateRuntimeOutput();
}

async function runInferenceTick(now, threshold) {
  if (!state.isRunning || !state.objectModel || state.inferenceBusy) return;
  if (now - state.lastInferenceAt < INFERENCE_TICK_INTERVAL_MS) return;

  state.inferenceBusy = true;
  state.lastInferenceAt = now;
  const runEpoch = state.runEpoch;

  try {
    try {
      const objectStartedAt = performance.now();
      const predictions = await withTimeout(
        state.objectModel.detect(video),
        OBJECT_INFERENCE_TIMEOUT_MS,
        "Object detection timeout"
      );

      if (!state.isRunning || runEpoch !== state.runEpoch) return;

      state.objectLatencyMs = performance.now() - objectStartedAt;
      state.objectErrors = 0;
      state.latestObjects = predictions
        .filter((pred) => pred.score >= threshold)
        .sort((a, b) => b.score - a.score);
      updateObjectList(state.latestObjects);

      if (state.latestObjects.length) {
        const top = state.latestObjects[0];
        setModelOutput(
          objectModelOutput,
          `${state.latestObjects.length} obj | ${top.class} ${(top.score * 100).toFixed(0)}%`,
          "ok"
        );
      } else {
        setModelOutput(
          objectModelOutput,
          `0 obj | ${state.objectLatencyMs.toFixed(0)}ms`,
          "warn"
        );
      }
    } catch (error) {
      console.error(error);
      state.objectErrors += 1;
      setModelOutput(objectModelOutput, "Inference error", "error");
      if (state.objectErrors >= 3) {
        setStatus("Object detection errors detected. Try Stop -> Start camera.", "error");
      }
      return;
    }

    const shouldRunFace =
      state.emotionReady && now - state.lastFaceInferAt >= FACE_DETECT_INTERVAL_MS;
    if (!shouldRunFace) return;

    state.lastFaceInferAt = now;

    try {
      const faceStartedAt = performance.now();
      const detections = await withTimeout(
        faceapi
          .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions(FACE_DETECT_OPTIONS))
          .withFaceExpressions(),
        FACE_INFERENCE_TIMEOUT_MS,
        "Face detection timeout"
      );

      if (!state.isRunning || runEpoch !== state.runEpoch) return;

      state.faceLatencyMs = performance.now() - faceStartedAt;
      state.emotionFailures = 0;
      state.latestFaces = detections
        .map((det) => {
          const topExpression = getTopExpression(det.expressions);
          const { x, y, width, height } = det.detection.box;
          return {
            bbox: [x, y, width, height],
            faceScore: det.detection.score || 0,
            emotion: topExpression.label,
            emotionScore: topExpression.score,
          };
        })
        .sort((a, b) => b.faceScore - a.faceScore);

      updateEmotionList(state.latestFaces);

      if (state.latestFaces.length) {
        const topFace = state.latestFaces[0];
        setModelOutput(
          faceDetectorOutput,
          `${state.latestFaces.length} face | ${(topFace.faceScore * 100).toFixed(0)}%`,
          "ok"
        );
        setModelOutput(
          expressionModelOutput,
          `${topFace.emotion} ${(topFace.emotionScore * 100).toFixed(0)}%`,
          topFace.emotionScore >= 0.42 ? "ok" : "warn"
        );
      } else {
        setModelOutput(
          faceDetectorOutput,
          `0 face | ${state.faceLatencyMs.toFixed(0)}ms`,
          "warn"
        );
        setModelOutput(expressionModelOutput, "No face", "warn");
      }
    } catch (error) {
      console.error(error);
      state.emotionFailures += 1;
      state.latestFaces = [];
      updateEmotionList([]);
      const faceError =
        typeof error?.message === "string"
          ? error.message.slice(0, 64)
          : "unknown";
      setModelOutput(faceDetectorOutput, `Face error: ${faceError}`, "error");
      setModelOutput(expressionModelOutput, "Emotion error", "error");

      if (state.emotionFailures >= 3) {
        state.emotionReady = false;
        if (!state.warnedEmotionFailure) {
          state.warnedEmotionFailure = true;
          setStatus(
            "Emotion detection failed. Running object-only. Check model status on the right.",
            "warn"
          );
        }
      }
    }
  } finally {
    if (runEpoch === state.runEpoch) {
      state.inferenceBusy = false;
    }
  }
}

function detectLoop(frameAt) {
  if (!state.isRunning) return;

  if (video.readyState < 2) {
    state.animationId = requestAnimationFrame(detectLoop);
    return;
  }

  resizeOverlayToVideo();

  if (state.lastFrameAt) {
    const frameDelta = Math.max(1, frameAt - state.lastFrameAt);
    const instantFps = 1000 / frameDelta;
    state.fps = state.fps ? state.fps * 0.88 + instantFps * 0.12 : instantFps;
  }
  state.lastFrameAt = frameAt;

  const threshold = Number(thresholdRange.value);
  runInferenceTick(frameAt, threshold);
  drawDetections(state.latestObjects, state.latestFaces);

  if (frameAt - state.lastRuntimeUpdateAt > 300) {
    state.lastRuntimeUpdateAt = frameAt;
    updateRuntimeOutput();
  }

  state.animationId = requestAnimationFrame(detectLoop);
}

thresholdRange.addEventListener("input", () => {
  thresholdValue.textContent = Number(thresholdRange.value).toFixed(2);
});

startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", stopCamera);
window.addEventListener("beforeunload", stopCamera);
window.addEventListener("resize", resizeOverlayToVideo);

if (!ctx) {
  setStatus("Canvas 2D is unavailable in this browser.", "error");
  startButton.disabled = true;
} else if (!("mediaDevices" in navigator)) {
  setStatus("Browser does not support webcam APIs.", "error");
  startButton.disabled = true;
} else {
  thresholdValue.textContent = Number(thresholdRange.value).toFixed(2);
  loadModels();
}

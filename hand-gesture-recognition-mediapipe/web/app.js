const PREDICT_INTERVAL_MS = 160;
const CAPTURE_WIDTH = 480;
const JPEG_QUALITY = 0.72;

const video = document.getElementById('video');
const videoEmpty = document.getElementById('videoEmpty');
const captureCanvas = document.getElementById('captureCanvas');
const connectionState = document.getElementById('connectionState');
const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
const resetButton = document.getElementById('resetButton');
const copyButton = document.getElementById('copyButton');
const backspaceButton = document.getElementById('backspaceButton');
const clearButton = document.getElementById('clearButton');
const outputText = document.getElementById('outputText');
const candidateLabel = document.getElementById('candidateLabel');
const candidateSource = document.getElementById('candidateSource');
const holdTimer = document.getElementById('holdTimer');
const cooldownTimer = document.getElementById('cooldownTimer');
const holdProgress = document.getElementById('holdProgress');
const keypointLabel = document.getElementById('keypointLabel');
const motionLabel = document.getElementById('motionLabel');
const lastCommit = document.getElementById('lastCommit');
const commitFlash = document.getElementById('commitFlash');

let stream = null;
let predictTimer = null;
let requestInFlight = false;
let flashTimer = null;

function setLiveState(isLive, label) {
  connectionState.textContent = label;
  connectionState.classList.toggle('is-live', isLive);
}

function formatLabel(value) {
  return value && value.trim() ? value : '-';
}

function formatSource(value) {
  if (value === 'keypoint') {
    return 'Tĩnh';
  }
  if (value === 'motion') {
    return 'Chuyển động';
  }
  return '-';
}

function renderState(state) {
  if (state.error) {
    setLiveState(false, state.error);
    return;
  }

  outputText.value = state.text || '';
  outputText.scrollTop = outputText.scrollHeight;

  const holdSeconds = Number(state.hold_seconds || 2);
  const elapsed = Math.min(Number(state.candidate_elapsed || 0), holdSeconds);
  const progress = holdSeconds > 0 ? Math.min(elapsed / holdSeconds, 1) : 0;
  const cooldown = Math.max(Number(state.cooldown_remaining || 0), 0);

  candidateLabel.textContent = formatLabel(state.candidate_label);
  candidateSource.textContent = formatSource(state.candidate_source);
  holdTimer.textContent = `${elapsed.toFixed(1)} / ${holdSeconds.toFixed(1)}s`;
  cooldownTimer.textContent = `${cooldown.toFixed(1)}s`;
  holdProgress.style.width = `${Math.round(progress * 100)}%`;
  keypointLabel.textContent = formatLabel(state.keypoint_label);
  motionLabel.textContent = formatLabel(state.motion_label);
  lastCommit.textContent = formatLabel(state.last_commit);

  if (state.committed) {
    showCommit(state.committed);
  }
}

function showCommit(label) {
  let displayLabel = label;
  if (label === 'Space') {
    displayLabel = 'Space';
  }
  commitFlash.textContent = displayLabel;
  commitFlash.classList.add('is-visible');

  window.clearTimeout(flashTimer);
  flashTimer = window.setTimeout(() => {
    commitFlash.classList.remove('is-visible');
  }, 700);
}

async function startCamera() {
  if (stream) {
    return;
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 960 },
        height: { ideal: 540 },
        facingMode: 'user',
      },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    videoEmpty.classList.add('is-hidden');
    startButton.disabled = true;
    stopButton.disabled = false;
    setLiveState(true, 'Đang nhận diện');
    predictTimer = window.setInterval(sendFrame, PREDICT_INTERVAL_MS);
  } catch (error) {
    setLiveState(false, 'Không mở được camera');
  }
}

function stopCamera() {
  window.clearInterval(predictTimer);
  predictTimer = null;

  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }

  video.srcObject = null;
  videoEmpty.classList.remove('is-hidden');
  startButton.disabled = false;
  stopButton.disabled = true;
  setLiveState(false, 'Đã tắt camera');
}

async function sendFrame() {
  if (!stream || requestInFlight || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    return;
  }

  requestInFlight = true;
  try {
    const image = captureFrame();
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ image }),
    });
    const state = await response.json();
    renderState(state);
  } catch (error) {
    setLiveState(false, 'Mất kết nối server');
  } finally {
    requestInFlight = false;
  }
}

function captureFrame() {
  const ratio = video.videoHeight / video.videoWidth || 9 / 16;
  captureCanvas.width = CAPTURE_WIDTH;
  captureCanvas.height = Math.round(CAPTURE_WIDTH * ratio);

  const context = captureCanvas.getContext('2d', { willReadFrequently: true });
  context.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
  return captureCanvas.toDataURL('image/jpeg', JPEG_QUALITY);
}

async function postControl(action) {
  try {
    const response = await fetch('/api/control', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ action }),
    });
    const state = await response.json();
    renderState(state);
  } catch (error) {
    setLiveState(false, 'Không gửi được lệnh');
  }
}

async function copyText() {
  try {
    await navigator.clipboard.writeText(outputText.value);
    setLiveState(Boolean(stream), stream ? 'Đã copy' : 'Đã copy văn bản');
  } catch (error) {
    outputText.focus();
    outputText.select();
    document.execCommand('copy');
  }
}

startButton.addEventListener('click', startCamera);
stopButton.addEventListener('click', stopCamera);
resetButton.addEventListener('click', () => postControl('reset'));
copyButton.addEventListener('click', copyText);
backspaceButton.addEventListener('click', () => postControl('backspace'));
clearButton.addEventListener('click', () => postControl('clear'));

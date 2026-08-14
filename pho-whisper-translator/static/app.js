const sourceText = document.querySelector("#source-text");
const translatedText = document.querySelector("#translated-text");
const resultPlaceholder = document.querySelector("#result-placeholder");
const characterCount = document.querySelector("#character-count");
const translateButton = document.querySelector("#translate-button");
const recordButton = document.querySelector("#record-button");
const recordLabel = document.querySelector("#record-label");
const listenButton = document.querySelector("#listen-button");
const copyButton = document.querySelector("#copy-button");
const statusElement = document.querySelector("#status");
const audioPlayer = document.querySelector("#audio-player");

const maxTextLength = Number(sourceText.maxLength);
const maxRecordingSeconds = 10;
let mediaRecorder = null;
let mediaStream = null;
let audioChunks = [];
let recordingTimer = null;
let recordingSecondsLeft = maxRecordingSeconds;
let activeAudioUrl = null;

function setStatus(message, state = "ready") {
  statusElement.className = `status ${state === "ready" ? "" : state}`.trim();
  statusElement.querySelector("span:last-child").textContent = message;
}

function setBusy(isBusy) {
  translateButton.disabled = isBusy;
  sourceText.disabled = isBusy;
  if (!mediaRecorder || mediaRecorder.state === "inactive") {
    recordButton.disabled = isBusy;
  }
}

function showTranslation(text) {
  translatedText.textContent = text;
  translatedText.classList.toggle("visible", Boolean(text));
  resultPlaceholder.classList.toggle("hidden", Boolean(text));
  listenButton.disabled = !text;
  copyButton.disabled = !text;
}

async function parseResponse(response) {
  if (response.ok) {
    return response.json();
  }
  let message = "Không thể kết nối tới máy chủ.";
  try {
    const body = await response.json();
    message = body.error || message;
  } catch (_error) {
    // Keep the generic message when the server does not return JSON.
  }
  throw new Error(message);
}

async function translateSource() {
  const text = sourceText.value.trim();
  if (!text) {
    setStatus("Vui lòng nhập nội dung tiếng Việt", "error");
    sourceText.focus();
    return;
  }

  setBusy(true);
  setStatus("Đang dịch sang tiếng Anh...", "busy");
  try {
    const response = await fetch("/api/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await parseResponse(response);
    showTranslation(data.translated_text);
    setStatus("Đã dịch xong");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setBusy(false);
  }
}

function preferredAudioType() {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/ogg;codecs=opus",
    "audio/mp4",
  ];
  return candidates.find((type) => MediaRecorder.isTypeSupported(type)) || "";
}

function extensionForMimeType(mimeType) {
  if (mimeType.includes("ogg")) return "ogg";
  if (mimeType.includes("mp4")) return "mp4";
  return "webm";
}

function updateRecordButton() {
  recordButton.classList.add("recording");
  recordButton.setAttribute("aria-label", "Dừng ghi âm");
  recordLabel.textContent = `Dừng · ${recordingSecondsLeft}s`;
}

function resetRecordButton() {
  clearInterval(recordingTimer);
  recordingTimer = null;
  recordButton.classList.remove("recording");
  recordButton.setAttribute("aria-label", "Bắt đầu ghi âm");
  recordLabel.textContent = "Ghi âm";
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
}

async function startRecording() {
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    setStatus("Trình duyệt không hỗ trợ ghi âm", "error");
    return;
  }

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
      },
    });

    audioChunks = [];
    const mimeType = preferredAudioType();
    mediaRecorder = new MediaRecorder(
      mediaStream,
      mimeType ? { mimeType } : undefined,
    );

    mediaRecorder.addEventListener("dataavailable", (event) => {
      if (event.data.size > 0) audioChunks.push(event.data);
    });
    mediaRecorder.addEventListener("stop", uploadRecording, { once: true });
    mediaRecorder.start();

    recordingSecondsLeft = maxRecordingSeconds;
    updateRecordButton();
    setStatus("Đang ghi âm...", "busy");
    recordingTimer = setInterval(() => {
      recordingSecondsLeft -= 1;
      updateRecordButton();
      if (recordingSecondsLeft <= 0) stopRecording();
    }, 1000);
  } catch (error) {
    resetRecordButton();
    setStatus(
      error.name === "NotAllowedError"
        ? "Bạn chưa cấp quyền sử dụng microphone"
        : "Không thể mở microphone",
      "error",
    );
  }
}

async function uploadRecording() {
  resetRecordButton();
  mediaStream?.getTracks().forEach((track) => track.stop());
  mediaStream = null;

  const mimeType = mediaRecorder.mimeType || "audio/webm";
  const audioBlob = new Blob(audioChunks, { type: mimeType });
  const formData = new FormData();
  formData.append(
    "audio",
    audioBlob,
    `recording.${extensionForMimeType(mimeType)}`,
  );

  setBusy(true);
  setStatus("PhoWhisper đang nhận dạng giọng nói...", "busy");
  try {
    const response = await fetch("/api/transcribe", {
      method: "POST",
      body: formData,
    });
    const data = await parseResponse(response);
    sourceText.value = data.source_text;
    sourceText.dispatchEvent(new Event("input"));
    showTranslation(data.translated_text);
    setStatus("Đã nhận dạng và dịch xong");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    mediaRecorder = null;
    audioChunks = [];
    setBusy(false);
  }
}

async function toggleRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    stopRecording();
  } else {
    await startRecording();
  }
}

async function listenToTranslation() {
  const text = translatedText.textContent.trim();
  if (!text) return;

  listenButton.disabled = true;
  setStatus("Đang tạo giọng đọc tiếng Anh...", "busy");
  try {
    const response = await fetch("/api/speech", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!response.ok) {
      await parseResponse(response);
    }
    const audioBlob = await response.blob();
    if (activeAudioUrl) URL.revokeObjectURL(activeAudioUrl);
    activeAudioUrl = URL.createObjectURL(audioBlob);
    audioPlayer.src = activeAudioUrl;
    await audioPlayer.play();
    setStatus("Đang phát bản dịch");
  } catch (error) {
    setStatus(error.message || "Không thể phát bản dịch", "error");
  } finally {
    listenButton.disabled = false;
  }
}

async function copyTranslation() {
  try {
    await navigator.clipboard.writeText(translatedText.textContent);
    setStatus("Đã sao chép bản dịch");
  } catch (_error) {
    setStatus("Không thể sao chép bản dịch", "error");
  }
}

sourceText.addEventListener("input", () => {
  characterCount.textContent = `${sourceText.value.length} / ${maxTextLength}`;
});
sourceText.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    translateSource();
  }
});
translateButton.addEventListener("click", translateSource);
recordButton.addEventListener("click", toggleRecording);
listenButton.addEventListener("click", listenToTranslation);
copyButton.addEventListener("click", copyTranslation);
audioPlayer.addEventListener("ended", () => setStatus("Đã phát xong"));

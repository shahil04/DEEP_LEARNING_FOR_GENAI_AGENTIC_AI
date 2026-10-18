// static/js/camera.js
(function() {
  const video = document.getElementById('video');
  const statusEl = document.getElementById('status');
  const faultsEl = document.getElementById('faults');
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');

  // Access webcam
  async function startWebcam() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      video.srcObject = stream;
      await video.play();
      // start sending frames
      setTimeout(sendFrameLoop, 1500); // slight delay before first capture
    } catch (err) {
      statusEl.innerText = 'Could not access webcam: ' + err;
    }
  }

  async function sendFrameLoop() {
    if (video.readyState < 2) {
      setTimeout(sendFrameLoop, 500);
      return;
    }
    // capture frame
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.8);

    try {
      const resp = await fetch('/monitor', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frame: dataUrl })
      });
      const result = await resp.json();
      if (result.error) {
        statusEl.innerText = 'Error: ' + result.error;
      } else {
        statusEl.innerText = 'Status: ' + result.status;
        faultsEl.innerText = 'Faults: ' + result.faults + ' / ' + result.max_faults;
        if (result.submit) {
          alert('Too many faults detected. Exam will be auto-submitted.');
          document.getElementById('examForm').submit();
          return;
        }
      }
    } catch (e) {
      console.error('monitor request failed', e);
      statusEl.innerText = 'Monitor request failed';
    }

    // schedule next capture
    setTimeout(sendFrameLoop, 1500);
  }

  startWebcam();
})();

const hands = new Hands({
  locateFile: (file) => {
    return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
  },
});
hands.setOptions({
  maxNumHands: 2,
  modelComplexity: 1,
  minDetectionConfidence: 0.69,
  minTrackingConfidence: 0.69,
});

export function initializeCamera() {
  const canvasElement = document.getElementsByClassName("output_canvas")[0];
  const canvasCtx = canvasElement.getContext("2d");

  // draw stuff on the image
  // TODO: calculate degree here i think
  function onResults(results) {
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.translate(480, 0);
    canvasCtx.scale(-1, 1);
    canvasCtx.drawImage(
      results.image,
      0,
      0,
      canvasElement.width,
      canvasElement.height
    );

    let pts = [];
    if (results.multiHandLandmarks) {
      for (const landmarks of results.multiHandLandmarks) {
        let landmark = landmarks[5];
        let x = parseInt(landmark.x * 480, 10); // width of camera
        let y = parseInt(landmark.y * 320, 10); // height
        pts.push([x, y]);
        for (const pt of pts) {
          console.log(pt);
          canvasCtx.beginPath();
          canvasCtx.arc(pt[0], pt[1], 5, 0, Math.PI * 2);
          canvasCtx.fillStyle = "red";
          canvasCtx.fill();
        }

        // move lower x coordinate pt to the left
        pts.sort((a, b) => {
          return a[0] - b[0];
        });
        let angle_deg = 0;
        let angle = 0;

        if (pts.length == 2) {
          // line between the two dots
          canvasCtx.beginPath();
          canvasCtx.moveTo(pts[0][0], pts[0][1]);
          canvasCtx.lineTo(pts[1][0], pts[1][1]);
          canvasCtx.lineWidth = 2;
          canvasCtx.strokeStyle = "red";
          canvasCtx.stroke();

          // horizontal line from left hand
          canvasCtx.beginPath();
          canvasCtx.moveTo(pts[1][0], pts[1][1]);
          canvasCtx.lineTo(0, pts[1][1]);
          canvasCtx.lineWidth = 2;
          canvasCtx.strokeStyle = "green";
          canvasCtx.stroke();

          // calculate angle
          let opposite = pts[1][1] - pts[0][1];
          let adjacent = pts[1][0] - pts[0][0];

          try {
            angle = Math.atan(opposite / adjacent);
          } catch (error) {
            // user's hands are stacked
          } finally {
            angle_deg = (angle * 180) / Math.PI;
          }

          canvasCtx.restore(); // undoing the flip from earlier
          canvasCtx.font = "30px sans-serif";
          canvasCtx.fillStyle = "red";
          canvasCtx.fillText(parseInt(angle_deg) + " deg", 40, 40);
        }
      }
    }
    canvasCtx.restore();
  }

  hands.onResults(onResults);
}

export const camera = (videoElement) =>
  new Camera(videoElement, {
    onFrame: async () => {
      await hands.send({ image: videoElement });
    },
    width: 320,
    height: 240,
  });

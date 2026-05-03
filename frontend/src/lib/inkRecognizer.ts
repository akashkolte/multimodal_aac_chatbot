const CANVAS_SIZE = 200;
const PAD = 20;

function renderStroke(trajectory: [number, number][]): string {
  const canvas = document.createElement("canvas");
  canvas.width = CANVAS_SIZE;
  canvas.height = CANVAS_SIZE;
  const ctx = canvas.getContext("2d")!;

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const [x, y] of trajectory) {
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
  }

  const drawArea = CANVAS_SIZE - 2 * PAD;
  const rangeX = Math.max(maxX - minX, 1);
  const rangeY = Math.max(maxY - minY, 1);
  const scale = Math.min(drawArea / rangeX, drawArea / rangeY);
  const offX = PAD + (drawArea - rangeX * scale) / 2;
  const offY = PAD + (drawArea - rangeY * scale) / 2;

  ctx.strokeStyle = "#000000";
  ctx.lineWidth = 5;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  const [x0, y0] = trajectory[0];
  ctx.moveTo((x0 - minX) * scale + offX, (y0 - minY) * scale + offY);
  for (let i = 1; i < trajectory.length; i++) {
    const [x, y] = trajectory[i];
    ctx.lineTo((x - minX) * scale + offX, (y - minY) * scale + offY);
  }
  ctx.stroke();

  return canvas.toDataURL("image/png").split(",")[1];
}

export async function recognizeInkStroke(
  trajectory: [number, number][]
): Promise<string> {
  if (trajectory.length < 5) return "";
  const base64 = renderStroke(trajectory);
  try {
    const res = await fetch("/ink/recognize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_base64: base64 }),
    });
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      console.error("[inkRecognizer] HTTP", res.status, body);
      return "";
    }
    const data = await res.json();
    const text = (data.text as string) || "";
    console.debug("[inkRecognizer] recognized:", JSON.stringify(text));
    return text;
  } catch (err) {
    console.error("[inkRecognizer] fetch failed:", err);
    return "";
  }
}

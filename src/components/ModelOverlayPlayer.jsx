import React, { useRef, useEffect, useState, useMemo, useCallback } from 'react';

/**
 * Phát lại video upload + vẽ skeleton (khung xương) + bbox + điểm nghi ngờ
 * theo TỪNG FRAME, dùng timeline keypoint do backend KLTN trả về (YOLO-Pose).
 * Điểm và keypoint là THẬT từ model — không tính lại trong trình duyệt.
 */

// COCO-17 skeleton edges
const EDGES = [
  [0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
];

export default function ModelOverlayPlayer({ videoUrl, fps, timeline, threshold = 0.5 }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const rafRef = useRef(null);
  const [cur, setCur] = useState({ score: null, frame: 0 });
  const [videoErr, setVideoErr] = useState(false);

  // index timeline by frame number for O(1) lookup
  const byFrame = useMemo(() => {
    const m = new Map();
    (timeline || []).forEach((e) => m.set(e.f, e));
    return m;
  }, [timeline]);

  const draw = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    const rect = video.getBoundingClientRect();
    if (canvas.width !== Math.round(rect.width) || canvas.height !== Math.round(rect.height)) {
      canvas.width = Math.round(rect.width);
      canvas.height = Math.round(rect.height);
    }
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const frame = Math.round(video.currentTime * (fps || 30));
    // nearest available timeline entry (within a few frames)
    let e = byFrame.get(frame);
    if (!e) {
      for (let d = 1; d <= 4 && !e; d++) e = byFrame.get(frame - d) || byFrame.get(frame + d);
    }
    const W = canvas.width, H = canvas.height;

    if (e && e.kp) {
      const kp = e.kp;
      const score = e.s;
      const alert = score != null && score >= threshold;
      const color = alert ? '#ef4444' : '#22c55e';

      // bbox from keypoints
      const valid = kp.filter((p) => !(p[0] === 0 && p[1] === 0));
      if (valid.length) {
        const xs = valid.map((p) => p[0]), ys = valid.map((p) => p[1]);
        const x1 = Math.min(...xs) * W, x2 = Math.max(...xs) * W;
        const y1 = Math.min(...ys) * H, y2 = Math.max(...ys) * H;
        ctx.strokeStyle = color; ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      }

      // skeleton edges
      ctx.strokeStyle = color; ctx.lineWidth = 2;
      EDGES.forEach(([a, b]) => {
        const pa = kp[a], pb = kp[b];
        if (pa && pb && !(pa[0] === 0 && pa[1] === 0) && !(pb[0] === 0 && pb[1] === 0)) {
          ctx.beginPath();
          ctx.moveTo(pa[0] * W, pa[1] * H);
          ctx.lineTo(pb[0] * W, pb[1] * H);
          ctx.stroke();
        }
      });
      // joints
      ctx.fillStyle = '#fde047';
      kp.forEach((p) => {
        if (!(p[0] === 0 && p[1] === 0)) {
          ctx.beginPath(); ctx.arc(p[0] * W, p[1] * H, 3, 0, 7); ctx.fill();
        }
      });

      setCur({ score, frame });
    } else {
      setCur((c) => ({ ...c, frame }));
    }
  }, [byFrame, fps, threshold]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return undefined;
    const loop = () => { draw(); rafRef.current = requestAnimationFrame(loop); };
    const start = () => { cancelAnimationFrame(rafRef.current); rafRef.current = requestAnimationFrame(loop); };
    const stop = () => cancelAnimationFrame(rafRef.current);
    video.addEventListener('play', start);
    video.addEventListener('pause', stop);
    video.addEventListener('seeked', draw);
    video.addEventListener('loadeddata', draw);
    return () => {
      stop();
      video.removeEventListener('play', start);
      video.removeEventListener('pause', stop);
      video.removeEventListener('seeked', draw);
      video.removeEventListener('loadeddata', draw);
    };
  }, [draw]);

  const pct = cur.score != null ? Math.round(cur.score * 100) : null;
  const alert = cur.score != null && cur.score >= threshold;

  return (
    <div className="mt-3">
      <div className="relative inline-block w-full bg-black rounded-lg overflow-hidden">
        <video
          ref={videoRef}
          src={videoUrl}
          controls
          autoPlay
          muted
          loop
          playsInline
          preload="auto"
          onError={() => setVideoErr(true)}
          className="w-full block"
          style={{ objectFit: 'fill', maxHeight: '55vh' }}
        />
        <canvas ref={canvasRef} className="absolute top-0 left-0 w-full h-full pointer-events-none" />
        {videoErr && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/80 text-center p-4">
            <div className="text-sm text-white">
              ⚠ Trình duyệt không phát được định dạng video này (có thể là codec H.265/HEVC).<br />
              Kết quả phân tích phía trên vẫn chính xác. Hãy thử video <b>.mp4 (H.264)</b> để xem khung xương.
            </div>
          </div>
        )}
        <div className="absolute top-2 left-2 px-2 py-1 rounded text-xs bg-black/60 text-white">
          {pct != null
            ? <>điểm: <b className={alert ? 'text-red-400' : 'text-green-400'}>{pct}%</b> · frame {cur.frame}</>
            : <>đang chờ đủ 90 frame… · frame {cur.frame}</>}
        </div>
      </div>
      <p className="text-xs text-gray-400 mt-1">
        ▶ Bấm play để xem khung xương bám người + điểm chạy theo thời gian. Đỏ = nghi ngờ.
      </p>
    </div>
  );
}

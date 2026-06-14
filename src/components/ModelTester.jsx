import React, { useState } from 'react';
import ModelOverlayPlayer from './ModelOverlayPlayer';
import { useAnalysis } from '../context/AnalysisContext';

const API = 'http://localhost:8000';
const RED = '#C8102E';
const NAVY = '#13294b';
const MODEL = 'v1.4';

export default function ModelTester() {
  const { addAnalysis } = useAnalysis();
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState('');
  const [videoUrl, setVideoUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const onPick = (e) => {
    const f = e.target.files[0];
    setFile(f);
    setFileName(f?.name || '');
    setResult(null);
    setError(null);
    setVideoUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return f ? URL.createObjectURL(f) : null;
    });
  };

  const saveToStore = (data) => {
    const events = data.events || [];
    addAnalysis({
      id: `kltn-${Date.now()}`,
      fileName: fileName || 'video',
      timestamp: new Date().toISOString(),
      model: 'V1.4',
      suspicious_count: events.length,
      // Mỗi sự kiện = 1 dòng cảnh báo (ảnh + số liệu riêng).
      suspicious_activities: events.map((e) => ({
        type: 'shoplifting',
        timestamp: e.peak_time,
        frame: e.peak_frame,
        confidence: e.score,
        details: `Sự kiện ${e.index}: nghi ngờ cất giấu hàng hóa (đoạn ${e.start_time}s–${e.end_time}s)`,
        image: e.snapshot_url ? `${API}${e.snapshot_url}` : null,
      })),
      people_count: data.n_person_frames > 0 ? 1 : 0,
      frame_count: data.n_frames,
      fps: data.fps || 30,
      avg_confidence: data.score_max ?? data.score,
    });
  };

  const run = async () => {
    if (!file) { setError('Hãy chọn video trước.'); return; }
    setLoading(true); setError(null); setResult(null);
    try {
      const fd = new FormData();
      fd.append('file', file);
      fd.append('model', MODEL);
      const res = await fetch(`${API}/analyze-model`, { method: 'POST', body: fd });
      if (!res.ok) throw new Error(`Lỗi máy chủ (${res.status})`);
      const data = await res.json();
      setResult(data);
      if (data.label !== 'no_person') saveToStore(data);
    } catch (e) {
      setError('Không kết nối được máy chủ phân tích (cổng 8000). Hãy chạy backend. — ' + (e.message || ''));
    } finally {
      setLoading(false);
    }
  };

  const isShop = result && result.label === 'shoplifting';
  const peakPct = result ? Math.round((result.score_max ?? result.score) * 100) : 0;
  const meanPct = result ? Math.round(result.score * 100) : 0;
  const thrPct = result ? Math.round((result.threshold ?? 0.5) * 100) : 50;

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-bold" style={{ color: NAVY }}>Tải video để phân tích</h2>
          <p className="text-gray-500 text-sm">Định dạng MP4 / AVI / MOV. Mỗi video nên có một người trong khung hình.</p>
        </div>
        <span className="text-xs font-semibold px-3 py-1.5 rounded-full text-white" style={{ background: NAVY }}>
          Mô hình: V1.4 BiLSTM-CNN
        </span>
      </div>

      <div className="flex flex-wrap items-end gap-3 mb-4">
        <div className="flex-1 min-w-[240px]">
          <label className="block text-xs font-medium text-gray-500 mb-1">Video giám sát</label>
          <input
            type="file"
            accept="video/*"
            onChange={onPick}
            className="block w-full text-sm text-gray-500 file:mr-3 file:py-2 file:px-4
              file:rounded-full file:border-0 file:text-sm file:font-semibold
              file:bg-red-50 file:text-[#C8102E] hover:file:bg-red-100"
          />
        </div>
        <button
          onClick={run}
          disabled={loading || !file}
          className="px-5 py-2 rounded-lg text-white font-semibold disabled:opacity-50"
          style={{ background: RED }}
        >
          {loading ? 'Đang phân tích…' : 'Phân tích'}
        </button>
      </div>

      {loading && (
        <p className="text-sm" style={{ color: RED }}>
          ⏳ Đang trích khung xương (YOLOv8-Pose) và chạy mô hình… video dài có thể mất 30–60 giây.
        </p>
      )}

      {error && (
        <div className="text-sm text-red-700 bg-red-50 border border-red-200 rounded-lg p-3">{error}</div>
      )}

      {result && result.label !== 'no_person' && (
        <div className={`rounded-lg p-4 border ${isShop ? 'bg-red-50 border-red-200' : 'bg-green-50 border-green-200'}`}>
          <div className="flex items-center justify-between mb-2">
            <span className={`text-xl font-bold ${isShop ? 'text-[#C8102E]' : 'text-green-600'}`}>
              {result.verdict}
            </span>
            <span className="text-sm text-gray-500">{fileName}</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-4 overflow-hidden mb-1">
            <div className="h-4" style={{ width: `${peakPct}%`, background: isShop ? RED : '#16a34a' }} />
          </div>
          <p className="text-sm text-gray-700">
            {isShop && <><strong className="text-[#C8102E]">{result.n_events} sự kiện bất thường</strong> · </>}
            Đỉnh nghi ngờ: <strong>{peakPct}%</strong>
            {' · '}Trung bình: {meanPct}% — ngưỡng {thrPct}%
          </p>
          <p className="text-xs text-gray-500 mt-1">
            {result.n_frames} frame · {result.n_person_frames} frame có người · {result.n_windows} cửa sổ
          </p>

          {isShop && result.events?.length > 0 && (
            <ul className="text-xs text-gray-700 mt-2 space-y-0.5 border-l-2 border-red-200 pl-3">
              {result.events.map((e) => (
                <li key={e.index}>
                  <span className="font-medium text-[#C8102E]">Sự kiện {e.index}:</span>{' '}
                  {e.peak_time}s (frame {e.peak_frame}) — nghi ngờ {Math.round(e.score * 100)}%
                  {' '}· đoạn {e.start_time}s–{e.end_time}s
                </li>
              ))}
            </ul>
          )}
          {isShop && (
            <p className="text-xs text-gray-400 mt-1">
              Mỗi đoạn nghi ngờ (điểm ≥ {thrPct}%) được ghi nhận thành 1 sự kiện kèm ảnh bằng chứng ·
              gửi <strong>1 email duy nhất</strong> tổng hợp tất cả sự kiện.
            </p>
          )}
          {isShop && (
            <p className="text-xs mt-1" style={{ color: result.email_sent ? '#16a34a' : '#ea580c' }}>
              {result.email_sent
                ? `📧 Đã gửi email cảnh báo tới ${result.recipient_email}`
                : '📧 Cảnh báo được ghi nhận (email chưa cấu hình hoặc gửi lỗi)'}
            </p>
          )}

          {result.timeline && result.timeline.length > 0 && (
            <ModelOverlayPlayer
              videoUrl={result.playback_url ? `${API}${result.playback_url}` : videoUrl}
              fps={result.fps || 30}
              timeline={result.timeline}
            />
          )}
        </div>
      )}

      {result && result.label === 'no_person' && (
        <div className="text-sm text-orange-600 bg-orange-50 border border-orange-200 rounded-lg p-3">
          Không phát hiện người trong video → không thể chấm điểm.
        </div>
      )}
    </div>
  );
}

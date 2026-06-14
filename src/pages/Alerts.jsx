import React, { useState } from 'react';
import { useAnalysis, selectAlerts } from '../context/AnalysisContext';

const NAVY = '#13294b';
const TYPE_LABELS = {
  shoplifting: 'Cất giấu hàng hóa',
  concealment: 'Cất giấu hàng hóa',
};

function fmtDateTime(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  if (isNaN(d)) return '';
  const p = (n) => String(n).padStart(2, '0');
  return `${p(d.getDate())}/${p(d.getMonth() + 1)}/${d.getFullYear()} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

export default function Alerts() {
  const { analyses, clearAnalyses } = useAnalysis();
  const alerts = selectAlerts(analyses);
  const [preview, setPreview] = useState(null);

  return (
    <div className="p-8">
      <div className="flex items-end justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: NAVY }}>Cảnh báo</h1>
          <p className="text-gray-500">Các sự kiện hành vi bất thường phát hiện từ video đã phân tích.</p>
        </div>
        <button
          onClick={clearAnalyses}
          disabled={alerts.length === 0}
          className="px-4 py-2 rounded-lg text-white font-medium disabled:opacity-50"
          style={{ background: NAVY }}
        >
          Xóa tất cả
        </button>
      </div>

      <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
        <h2 className="text-lg font-bold mb-4">
          Cảnh báo gần đây <span className="text-gray-400 text-base">({alerts.length})</span>
        </h2>

        {alerts.length === 0 ? (
          <div className="text-center py-12 text-gray-400">
            Chưa có cảnh báo. Hãy phân tích một video ở trang{' '}
            <span className="font-medium">Phân tích video</span>.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead>
                <tr>
                  {['Ảnh', 'Video', 'Loại hành vi', 'Frame', 'Thời điểm', 'Độ tin cậy', 'Phát hiện'].map((h) => (
                    <th key={h} className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-100">
                {alerts.map((a) => (
                  <tr key={a.id}>
                    <td className="px-4 py-3">
                      {a.image ? (
                        <img
                          src={a.image}
                          alt="bằng chứng"
                          onClick={() => setPreview(a)}
                          className="h-14 w-20 object-cover rounded cursor-pointer border hover:ring-2 hover:ring-[#C8102E]"
                        />
                      ) : (
                        <span className="text-xs text-gray-400">—</span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-700">{a.fileName}</td>
                    <td className="px-4 py-3">
                      <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-red-100 text-[#C8102E]">
                        {TYPE_LABELS[a.type] || a.type}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-sm">{a.frame}</td>
                    <td className="px-4 py-3 text-sm">{(a.timestamp ?? 0).toFixed(1)}s</td>
                    <td className="px-4 py-3 text-sm font-semibold" style={{ color: NAVY }}>
                      {a.confidence != null ? `${Math.round(a.confidence * 100)}%` : '—'}
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-500 whitespace-nowrap">{fmtDateTime(a.at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {preview && (
        <div
          className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4"
          onClick={() => setPreview(null)}
        >
          <div className="bg-white rounded-lg overflow-hidden max-w-2xl w-full" onClick={(e) => e.stopPropagation()}>
            <img src={preview.image} alt="bằng chứng" className="w-full object-contain bg-black" />
            <div className="p-4">
              <p className="font-semibold text-[#C8102E]">{TYPE_LABELS[preview.type] || preview.type}</p>
              <p className="text-sm text-gray-600">{preview.details}</p>
              <p className="text-sm text-gray-500 mt-1">
                {preview.fileName} · frame {preview.frame} · {(preview.timestamp ?? 0).toFixed(1)}s ·{' '}
                {preview.confidence != null ? `${Math.round(preview.confidence * 100)}% tin cậy` : ''}
              </p>
              <button
                onClick={() => setPreview(null)}
                className="mt-3 px-4 py-2 rounded-lg text-white"
                style={{ background: NAVY }}
              >
                Đóng
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

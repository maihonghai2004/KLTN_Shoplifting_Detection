import React from 'react';
import { Bar, Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS, CategoryScale, LinearScale, BarElement, ArcElement,
  Title, Tooltip, Legend,
} from 'chart.js';
import { useAnalysis } from '../context/AnalysisContext';

ChartJS.register(CategoryScale, LinearScale, BarElement, ArcElement, Title, Tooltip, Legend);

const NAVY = '#13294b';
const RED = '#C8102E';

function Card({ children, className = '' }) {
  return <div className={`bg-white rounded-xl shadow-sm border border-gray-100 ${className}`}>{children}</div>;
}

function fmtDateTime(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  if (isNaN(d)) return '—';
  const p = (n) => String(n).padStart(2, '0');
  return `${p(d.getDate())}/${p(d.getMonth() + 1)} ${p(d.getHours())}:${p(d.getMinutes())}`;
}

function fmtDuration(sec) {
  if (!sec || sec < 0) return '—';
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return m > 0 ? `${m}m${String(s).padStart(2, '0')}s` : `${s}s`;
}

export default function Analytics() {
  const { analyses } = useAnalysis();

  // ----- Gom dữ liệu -----
  const events = analyses.flatMap((a) =>
    (a.suspicious_activities || []).map((ev) => ({ ...ev, fileName: a.fileName, at: a.timestamp })),
  );
  const flagged = analyses.filter((a) => (a.suspicious_activities?.length || 0) > 0).length;
  const totalFrames = analyses.reduce((s, a) => s + (a.frame_count || 0), 0);
  const totalDuration = analyses.reduce((s, a) => s + (a.frame_count || 0) / (a.fps || 30), 0);
  const confs = events.map((e) => e.confidence || 0);
  const avgConf = confs.length ? confs.reduce((s, c) => s + c, 0) / confs.length : 0;
  const flagRate = analyses.length ? Math.round((flagged / analyses.length) * 100) : 0;

  // ----- Phân tầng mức độ -----
  const sev = { high: 0, mid: 0, low: 0 };
  confs.forEach((c) => {
    if (c >= 0.75) sev.high += 1;
    else if (c >= 0.6) sev.mid += 1;
    else sev.low += 1;
  });

  // ----- Histogram độ tin cậy -----
  const bins = [0, 0, 0, 0, 0]; // 50-60, 60-70, 70-80, 80-90, 90-100
  confs.forEach((c) => {
    const i = Math.min(4, Math.max(0, Math.floor(c * 10) - 5));
    bins[i] += 1;
  });

  // ----- Insight tự động -----
  const critical = events.reduce((m, e) => ((e.confidence || 0) > (m?.confidence ?? -1) ? e : m), null);
  const borderline = confs.filter((c) => c < 0.6).length;
  const busiest = [...analyses]
    .map((a) => ({ f: a.fileName, n: a.suspicious_activities?.length || 0 }))
    .sort((x, y) => y.n - x.n)[0];
  const avgEvPerFlag = flagged ? (events.length / flagged).toFixed(1) : '0';

  const insights = [];
  insights.push({ t: 'info', m: `Tỉ lệ video bị gắn cờ: ${flagRate}% (${flagged}/${analyses.length} video).` });
  if (critical) {
    insights.push({
      t: 'alert',
      m: `Sự kiện đáng chú ý nhất: "${critical.fileName}" — tin cậy ${Math.round((critical.confidence || 0) * 100)}% tại ${Number(critical.timestamp ?? 0).toFixed(1)}s.`,
    });
  }
  if (busiest && busiest.n > 0) {
    insights.push({ t: 'info', m: `Video nhiều sự kiện nhất: "${busiest.f}" (${busiest.n} sự kiện). Trung bình ${avgEvPerFlag} sự kiện / video bị cờ.` });
  }
  if (borderline > 0) {
    insights.push({
      t: 'warn',
      m: `${borderline} sự kiện ở mức RANH GIỚI (<60%) — đề nghị người giám sát xem lại thủ công để tránh báo nhầm.`,
    });
  }
  if (sev.high > 0) {
    insights.push({ t: 'alert', m: `${sev.high} sự kiện độ tin cậy CAO (≥75%) — ưu tiên xử lý trước.` });
  }

  // ----- Chart configs -----
  const histData = {
    labels: ['50–60%', '60–70%', '70–80%', '80–90%', '90–100%'],
    datasets: [{
      label: 'Số sự kiện',
      data: bins,
      backgroundColor: ['#9ca3af', '#f59e0b', '#fb923c', '#ef4444', '#C8102E'],
    }],
  };
  const histOpts = {
    responsive: true,
    plugins: { legend: { display: false }, title: { display: true, text: 'Phân bố độ tin cậy của các sự kiện' } },
    scales: { y: { beginAtZero: true, ticks: { precision: 0 } } },
  };
  const sevData = {
    labels: ['Cao (≥75%)', 'Trung bình (60–75%)', 'Ranh giới (<60%)'],
    datasets: [{ data: [sev.high, sev.mid, sev.low], backgroundColor: ['#C8102E', '#f59e0b', '#9ca3af'] }],
  };
  const sevOpts = {
    responsive: true,
    plugins: { legend: { position: 'bottom' }, title: { display: true, text: 'Mức độ nghiêm trọng' } },
  };

  const rows = [...analyses]
    .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
    .map((a) => {
      const acts = a.suspicious_activities || [];
      const peak = acts.reduce((m, e) => Math.max(m, e.confidence || 0), a.avg_confidence || 0);
      return {
        id: a.id,
        fileName: a.fileName,
        at: a.timestamp,
        duration: (a.frame_count || 0) / (a.fps || 30),
        nEvents: acts.length,
        peak,
        flagged: acts.length > 0,
      };
    });

  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold mb-1" style={{ color: NAVY }}>Thống kê giám sát</h1>
      <p className="text-gray-500 mb-6">Báo cáo tổng hợp từ {analyses.length} video · {events.length} sự kiện bất thường.</p>

      {analyses.length === 0 ? (
        <Card className="p-12 text-center text-gray-400">
          Chưa có dữ liệu. Hãy phân tích một video ở trang <span className="font-medium">Phân tích video</span>.
        </Card>
      ) : (
        <>
          {/* KPI */}
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-6">
            <Kpi label="Video phân tích" value={analyses.length} />
            <Kpi label="Bị gắn cờ" value={`${flagged} (${flagRate}%)`} />
            <Kpi label="Tổng sự kiện" value={events.length} />
            <Kpi label="Độ tin cậy TB" value={`${Math.round(avgConf * 100)}%`} />
            <Kpi label="Thời lượng đã xử lý" value={fmtDuration(totalDuration)} />
            <Kpi label="Tổng frame" value={totalFrames.toLocaleString()} />
          </div>

          {/* Insights */}
          <Card className="p-6 mb-6">
            <h2 className="text-lg font-bold mb-3" style={{ color: NAVY }}>Nhận định cho người giám sát</h2>
            <ul className="space-y-2">
              {insights.map((it, i) => (
                <li key={i} className="flex items-start gap-2 text-sm">
                  <span>{it.t === 'alert' ? '🔴' : it.t === 'warn' ? '🟠' : 'ℹ️'}</span>
                  <span className="text-gray-700">{it.m}</span>
                </li>
              ))}
            </ul>
          </Card>

          {/* Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
            <Card className="p-6">
              <h2 className="text-lg font-bold mb-4" style={{ color: NAVY }}>Phân bố độ tin cậy</h2>
              {events.length > 0
                ? <Bar options={histOpts} data={histData} />
                : <p className="text-gray-400 py-12 text-center">Chưa có sự kiện bất thường.</p>}
            </Card>
            <Card className="p-6">
              <h2 className="text-lg font-bold mb-4" style={{ color: NAVY }}>Mức độ nghiêm trọng</h2>
              {events.length > 0
                ? <div className="max-w-xs mx-auto"><Doughnut options={sevOpts} data={sevData} /></div>
                : <p className="text-gray-400 py-12 text-center">Chưa có sự kiện bất thường.</p>}
            </Card>
          </div>

          {/* Bảng chi tiết theo video */}
          <Card className="p-6">
            <h2 className="text-lg font-bold mb-4" style={{ color: NAVY }}>Chi tiết theo video</h2>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead>
                  <tr>
                    {['Video', 'Thời gian phân tích', 'Thời lượng', 'Số sự kiện', 'Đỉnh tin cậy', 'Kết quả'].map((h) => (
                      <th key={h} className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {rows.map((r) => (
                    <tr key={r.id}>
                      <td className="px-4 py-2 text-sm text-gray-800">{r.fileName}</td>
                      <td className="px-4 py-2 text-sm text-gray-500 whitespace-nowrap">{fmtDateTime(r.at)}</td>
                      <td className="px-4 py-2 text-sm text-gray-600">{fmtDuration(r.duration)}</td>
                      <td className="px-4 py-2 text-sm font-medium" style={{ color: r.nEvents > 0 ? RED : '#16a34a' }}>{r.nEvents}</td>
                      <td className="px-4 py-2 text-sm font-semibold" style={{ color: NAVY }}>{Math.round(r.peak * 100)}%</td>
                      <td className="px-4 py-2">
                        {r.flagged ? (
                          <span className="px-2 inline-flex text-xs font-semibold rounded-full bg-red-100 text-[#C8102E]">Gắn cờ</span>
                        ) : (
                          <span className="px-2 inline-flex text-xs font-semibold rounded-full bg-green-100 text-green-700">Bình thường</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-xs text-gray-400 mt-3">Mô hình: V1.4 BiLSTM-CNN · ngưỡng cảnh báo 50% · độ tin cậy = điểm đỉnh của sự kiện.</p>
          </Card>
        </>
      )}
    </div>
  );
}

function Kpi({ label, value }) {
  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-4">
      <div className="text-xs text-gray-400">{label}</div>
      <div className="text-xl font-extrabold" style={{ color: RED }}>{value}</div>
    </div>
  );
}

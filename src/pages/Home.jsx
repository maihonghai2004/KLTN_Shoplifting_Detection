import React from 'react';
import { Link } from 'react-router-dom';
import Logo from '../components/Logo';

const RED = '#C8102E';
const NAVY = '#13294b';

function Card({ children, className = '' }) {
  return <div className={`bg-white rounded-xl shadow-sm border border-gray-100 ${className}`}>{children}</div>;
}

export default function Home() {
  return (
    <div>
      {/* University header band */}
      <div className="px-8 py-5 flex items-center gap-4" style={{ background: NAVY }}>
        <Logo size={56} />
        <div className="text-white leading-tight">
          <div className="text-sm tracking-wide text-blue-200">TRƯỜNG ĐẠI HỌC</div>
          <div className="text-lg font-bold">SƯ PHẠM KỸ THUẬT TP. HỒ CHÍ MINH</div>
          <div className="text-xs text-blue-200">HCMC University of Technology and Education</div>
        </div>
      </div>

      <div className="p-8 max-w-5xl mx-auto space-y-6">
        {/* Title */}
        <div className="text-center">
          <div className="inline-block px-3 py-1 rounded-full text-xs font-semibold text-white mb-3" style={{ background: RED }}>
            KHÓA LUẬN TỐT NGHIỆP · NGÀNH KỸ THUẬT DỮ LIỆU
          </div>
          <h1 className="text-3xl font-extrabold" style={{ color: NAVY }}>
            Ứng dụng học sâu nhận diện hành vi bất thường trong siêu thị
          </h1>
          <p className="text-gray-500 mt-2">
            Phát hiện hành vi cất giấu hàng hóa trái phép từ camera giám sát bằng phương pháp
            nhận dạng hành động dựa trên khung xương (skeleton-based action recognition).
          </p>
        </div>

        {/* Team */}
        <Card className="p-6">
          <div className="grid md:grid-cols-3 gap-4 text-sm">
            <div>
              <div className="text-gray-400 uppercase text-xs mb-1">Sinh viên thực hiện</div>
              <div className="font-semibold">Mai Hồng Hải <span className="text-gray-400">— 22133014</span></div>
              <div className="font-semibold">Nguyễn Ngọc Hiếu Hảo <span className="text-gray-400">— 22133015</span></div>
            </div>
            <div>
              <div className="text-gray-400 uppercase text-xs mb-1">Giảng viên hướng dẫn</div>
              <div className="font-semibold">ThS. Phan Thị Thể</div>
            </div>
            <div>
              <div className="text-gray-400 uppercase text-xs mb-1">Đơn vị</div>
              <div className="font-semibold">Khoa CNTT — HCMUTE</div>
            </div>
          </div>
        </Card>

        {/* Results */}
        <div className="grid md:grid-cols-4 gap-4">
          {[
            ['F1-macro', '0.807', 'Ensemble V2.3 (split-by-clip, 5 seed)'],
            ['Precision', '0.812', 'KPI ≥ 0.78 ✓'],
            ['Recall', '0.808', 'KPI ≥ 0.80 ✓'],
            ['Dữ liệu', '182', 'video gán nhãn (normal / shoplifting)'],
          ].map(([k, v, d]) => (
            <Card key={k} className="p-4">
              <div className="text-xs text-gray-400">{k}</div>
              <div className="text-2xl font-extrabold" style={{ color: RED }}>{v}</div>
              <div className="text-[11px] text-gray-500 mt-1">{d}</div>
            </Card>
          ))}
        </div>

        {/* Pipeline */}
        <Card className="p-6">
          <h2 className="text-lg font-bold mb-4" style={{ color: NAVY }}>Kiến trúc hệ thống</h2>
          <div className="flex flex-wrap items-center gap-2 text-sm">
            {['Video CCTV', 'YOLOv8-Pose (17 khớp)', 'Trích 56 đặc trưng hình học', 'Cửa sổ 90 frame', 'Mô hình học sâu (BiLSTM-CNN / ST-GCN)', 'Điểm nghi ngờ → Cảnh báo'].map((s, i, arr) => (
              <React.Fragment key={s}>
                <span className="px-3 py-2 rounded-lg bg-gray-50 border border-gray-200">{s}</span>
                {i < arr.length - 1 && <span className="text-gray-300">→</span>}
              </React.Fragment>
            ))}
          </div>
          <p className="text-sm text-gray-500 mt-4">
            Mô hình tốt nhất kết hợp <b>ST-GCN</b> (đồ thị không gian–thời gian của khung xương) với
            <b> BiLSTM-CNN</b>, đánh giá nghiêm ngặt theo <b>split-by-clip</b> để tránh rò rỉ dữ liệu.
          </p>
        </Card>

        {/* CTA */}
        <div className="text-center">
          <Link
            to="/demo"
            className="inline-block px-6 py-3 rounded-lg text-white font-semibold shadow-sm"
            style={{ background: RED }}
          >
            ▶ Vào trang Phân tích video
          </Link>
        </div>
      </div>
    </div>
  );
}

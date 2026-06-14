import React from 'react';
import { NavLink } from 'react-router-dom';
import { RiHome5Line, RiVideoUploadLine, RiAlarmWarningLine, RiBarChart2Line } from 'react-icons/ri';
import Logo from './Logo';

const nav = [
  { path: '/', icon: RiHome5Line, text: 'Giới thiệu', end: true },
  { path: '/demo', icon: RiVideoUploadLine, text: 'Phân tích video' },
  { path: '/alerts', icon: RiAlarmWarningLine, text: 'Cảnh báo' },
  { path: '/analytics', icon: RiBarChart2Line, text: 'Thống kê' },
];

export default function Sidebar() {
  return (
    <aside className="w-64 shrink-0 bg-[#13294b] text-white flex flex-col">
      {/* Brand */}
      <div className="px-4 py-5 border-b border-white/10 flex items-center gap-3">
        <Logo size={42} />
        <div className="leading-tight">
          <div className="text-[11px] uppercase tracking-wide text-blue-200">HCMUTE</div>
          <div className="text-sm font-bold">Nhận diện hành vi<br />bất thường</div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 p-3 space-y-1">
        {nav.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            end={item.end}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors ${
                isActive
                  ? 'bg-[#C8102E] text-white font-semibold'
                  : 'text-blue-100 hover:bg-white/10'
              }`
            }
          >
            <item.icon className="text-lg" />
            <span>{item.text}</span>
          </NavLink>
        ))}
      </nav>

      {/* Thesis footer */}
      <div className="p-4 border-t border-white/10 text-[11px] text-blue-200 leading-relaxed">
        <div className="font-semibold text-white mb-1">Khóa luận tốt nghiệp</div>
        <div>Ngành Kỹ thuật Dữ liệu</div>
        <div className="mt-1">Mai Hồng Hải · Nguyễn Ngọc Hiếu Hảo</div>
        <div>GVHD: ThS. Phan Thị Thể</div>
      </div>
    </aside>
  );
}

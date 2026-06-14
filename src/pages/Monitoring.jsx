import React from 'react';
import ModelTester from '../components/ModelTester';

const NAVY = '#13294b';

export default function Monitoring() {
  return (
    <div className="p-8 max-w-5xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-bold" style={{ color: NAVY }}>Phân tích video</h1>
        <p className="text-gray-500">
          Tải lên video giám sát → hệ thống trích khung xương (YOLOv8-Pose) và chạy mô hình học sâu
          để phát hiện hành vi cất giấu hàng hóa, hiển thị khung xương + điểm nghi ngờ theo thời gian.
        </p>
      </div>

      <ModelTester />
    </div>
  );
}

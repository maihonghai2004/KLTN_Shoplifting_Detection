import React, { useState } from 'react';

/**
 * Logo HCMUTE. Nếu có file public/hcmute-logo.png thì hiển thị ảnh thật;
 * nếu không, fallback sang badge chữ "HCMUTE" để giao diện luôn có thương hiệu.
 */
export default function Logo({ size = 44 }) {
  const [ok, setOk] = useState(true);
  if (ok) {
    return (
      <img
        src="/hcmute-logo.png"
        alt="HCMUTE"
        style={{ height: size }}
        className="object-contain bg-white rounded-md p-1"
        onError={() => setOk(false)}
      />
    );
  }
  return (
    <div
      className="flex items-center justify-center rounded-md bg-white text-[#C8102E] font-extrabold tracking-tight"
      style={{ height: size, width: size }}
    >
      <span className="text-sm">HCMUTE</span>
    </div>
  );
}

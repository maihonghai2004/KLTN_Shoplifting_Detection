import React, { createContext, useContext, useState, useCallback } from 'react';

/**
 * Shared store for video-analysis results.
 *
 * Every time a video is analysed we push a normalized "analysis" record here.
 * Pages like Alerts / Analytics / Dashboard read from this store so they show
 * REAL numbers from the backend instead of hard-coded demo data.
 *
 * The history (without the bulky per-frame data) is persisted to localStorage
 * so the stats survive a page reload. The raw `detections` array and the local
 * video URL are kept in memory only (they are large / session-bound).
 */

const STORAGE_KEY = 'shoplifting.analyses.v1';
const API = 'http://localhost:8000';
const AnalysisContext = createContext(null);

// Persist the full record. Screenshots are now stored on the backend as a small
// URL (e.g. /static/snap_xxx.jpg), not base64 — so keeping `image` in
// localStorage is cheap and the evidence survives reloads / future demos.
const toPersistable = (a) => ({
  id: a.id,
  fileName: a.fileName,
  timestamp: a.timestamp,
  frame_count: a.frame_count,
  fps: a.fps,
  suspicious_count: a.suspicious_count,
  people_count: a.people_count,
  avg_confidence: a.avg_confidence,
  suspicious_activities: a.suspicious_activities || [],
});

const loadPersisted = () => {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
};

export function AnalysisProvider({ children }) {
  // `analyses` holds the in-memory (full) records for the current session,
  // seeded from the persisted (lightweight) history.
  const [analyses, setAnalyses] = useState(loadPersisted);

  const persist = useCallback((list) => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(list.map(toPersistable)));
    } catch {
      /* ignore quota / serialization errors */
    }
  }, []);

  const addAnalysis = useCallback((record) => {
    setAnalyses((prev) => {
      const next = [record, ...prev].slice(0, 50); // keep last 50
      persist(next);
      return next;
    });
  }, [persist]);

  // Insert OR update a record by id (used for live, continuously-updating
  // analyses so Alerts/Analytics reflect events as they are detected).
  const upsertAnalysis = useCallback((record) => {
    setAnalyses((prev) => {
      const idx = prev.findIndex((a) => a.id === record.id);
      let next;
      if (idx === -1) {
        next = [record, ...prev].slice(0, 50);
      } else {
        next = [...prev];
        next[idx] = record;
      }
      persist(next);
      return next;
    });
  }, [persist]);

  const clearAnalyses = useCallback(() => {
    setAnalyses([]);
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {
      /* ignore */
    }
    // Đồng thời xóa ảnh bằng chứng + video phát lại đã lưu trên server.
    fetch(`${API}/clear-alerts`, { method: 'POST' }).catch(() => {});
  }, []);

  const value = {
    analyses,
    latest: analyses[0] || null,
    addAnalysis,
    upsertAnalysis,
    clearAnalyses,
  };

  return <AnalysisContext.Provider value={value}>{children}</AnalysisContext.Provider>;
}

export function useAnalysis() {
  const ctx = useContext(AnalysisContext);
  if (!ctx) {
    throw new Error('useAnalysis must be used within an AnalysisProvider');
  }
  return ctx;
}

/**
 * Flatten every analysis' suspicious_activities into a single list of alert
 * rows (most recent first), enriched with the source video + a stable id.
 */
export function selectAlerts(analyses) {
  const rows = [];
  analyses.forEach((a) => {
    (a.suspicious_activities || []).forEach((act, i) => {
      rows.push({
        id: `${a.id}-${i}`,
        analysisId: a.id,
        fileName: a.fileName,
        type: act.type,
        details: act.details,
        trackId: act.trackId,
        frame: act.frame,
        timestamp: act.timestamp,
        confidence: act.confidence,
        image: act.image || null,
        at: a.timestamp,
      });
    });
  });
  return rows;
}

/** Aggregate counts of each suspicious-activity type across all analyses. */
export function selectTypeCounts(analyses) {
  const counts = {};
  analyses.forEach((a) => {
    (a.suspicious_activities || []).forEach((act) => {
      counts[act.type] = (counts[act.type] || 0) + 1;
    });
  });
  return counts;
}

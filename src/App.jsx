import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Home from './pages/Home';
import Monitoring from './pages/Monitoring';
import Alerts from './pages/Alerts';
import Analytics from './pages/Analytics';
import { AnalysisProvider } from './context/AnalysisContext';

// Error boundary đơn giản
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }
  componentDidCatch(error, info) {
    console.error('UI Error:', error, info);
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-gray-100 p-8 text-center">
          <div>
            <h2 className="text-xl font-bold text-[#C8102E] mb-2">Đã xảy ra lỗi giao diện</h2>
            <p className="text-gray-600">{String(this.state.error?.message || this.state.error)}</p>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

export default function App() {
  return (
    <AnalysisProvider>
      <Router>
        <ErrorBoundary>
          <div className="flex h-screen bg-gray-100 text-gray-800">
            <Sidebar />
            <main className="flex-1 overflow-auto">
              <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/demo" element={<Monitoring />} />
                <Route path="/alerts" element={<Alerts />} />
                <Route path="/analytics" element={<Analytics />} />
                <Route path="*" element={<Navigate to="/" replace />} />
              </Routes>
            </main>
          </div>
        </ErrorBoundary>
      </Router>
    </AnalysisProvider>
  );
}

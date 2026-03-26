import React, { useState, useEffect, useCallback } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import Navbar from './components/Navbar';
import ProtectedRoute from './components/ProtectedRoute';
import LoginPage from './pages/LoginPage';
import SignupPage from './pages/SignupPage';
import StudentPracticePage from './pages/StudentPracticePage';
import TeacherDashboardPage from './pages/TeacherDashboardPage';
import config from './config';

function AppLayout({ children }) {
  const [theme, setTheme] = useState('dark');
  const { user, profile, signOut } = useAuth();

  // Theme management
  useEffect(() => {
    const savedTheme = localStorage.getItem(config.storageKeys.theme) || 'dark';
    setTheme(savedTheme);
    document.documentElement.setAttribute('data-theme', savedTheme);
  }, []);

  const toggleTheme = useCallback(() => {
    const newTheme = theme === 'dark' ? 'light' : 'dark';
    setTheme(newTheme);
    localStorage.setItem(config.storageKeys.theme, newTheme);
    document.documentElement.setAttribute('data-theme', newTheme);
  }, [theme]);

  return (
    <div className="app">
      <div className="bg-gradient"></div>
      <div className="bg-grid"></div>
      <div className="floating-orbs">
        <div className="orb orb-1"></div>
        <div className="orb orb-2"></div>
        <div className="orb orb-3"></div>
      </div>

      <Navbar
        theme={theme}
        onToggleTheme={toggleTheme}
        user={user}
        profile={profile}
        onSignOut={signOut}
      />

      <main className="main-content">
        {children}
      </main>

      <footer className="footer">
        <div className="footer-content">
          <div className="footer-brand">
            <i className="fas fa-graduation-cap"></i>
            <span>MAP Student Practice</span>
          </div>
          <div className="footer-info">
            <span>MCQ + explanation + misconception feedback</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

function HomeRedirect() {
  const { user, profile, isTeacher, loading } = useAuth();

  // Wait for auth to complete
  if (loading) {
    return <div className="auth-loading" />;
  }

  if (!user) {
    return <Navigate to="/login" replace />;
  }

  // Wait for profile to load (user exists but profile not yet fetched)
  if (!profile) {
    return <div className="auth-loading" />;
  }

  // Redirect based on role
  if (isTeacher) {
    return <Navigate to="/dashboard" replace />;
  }

  return <Navigate to="/practice" replace />;
}

function AppRoutes() {
  const { user } = useAuth();

  return (
    <Routes>
      {/* Public routes */}
      <Route
        path="/login"
        element={user ? <Navigate to="/" replace /> : <LoginPage />}
      />
      <Route
        path="/signup"
        element={user ? <Navigate to="/" replace /> : <SignupPage />}
      />

      {/* Home - redirects based on auth/role */}
      <Route path="/" element={<HomeRedirect />} />

      {/* Student practice */}
      <Route
        path="/practice"
        element={
          <ProtectedRoute>
            <StudentPracticePage />
          </ProtectedRoute>
        }
      />

      {/* Teacher dashboard */}
      <Route
        path="/dashboard"
        element={
          <ProtectedRoute requiredRole="teacher">
            <TeacherDashboardPage />
          </ProtectedRoute>
        }
      />

      {/* Fallback */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <AppLayout>
          <AppRoutes />
        </AppLayout>
      </AuthProvider>
    </BrowserRouter>
  );
}

export default App;

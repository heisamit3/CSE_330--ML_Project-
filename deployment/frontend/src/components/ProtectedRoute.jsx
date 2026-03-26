import { Navigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

function ProtectedRoute({ children, requiredRole }) {
  const { user, profile, loading, isTeacher } = useAuth();

  if (loading) {
    return <div className="auth-loading" />;
  }

  if (!user) {
    return <Navigate to="/login" replace />;
  }

  // Wait for profile to load before checking roles
  if (!profile) {
    return <div className="auth-loading" />;
  }

  // Check role requirement
  if (requiredRole === 'teacher' && !isTeacher) {
    return <Navigate to="/practice" replace />;
  }

  if (requiredRole === 'student' && isTeacher) {
    return <Navigate to="/dashboard" replace />;
  }

  return children;
}

export default ProtectedRoute;

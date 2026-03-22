import React from 'react';

function Toast({ message, type = 'success' }) {
  const icon = type === 'success' ? 'check-circle' : 'exclamation-circle';

  return (
    <div className={`toast toast-${type}`}>
      <i className={`toast-icon fas fa-${icon}`}></i>
      <span>{message}</span>
    </div>
  );
}

export default Toast;

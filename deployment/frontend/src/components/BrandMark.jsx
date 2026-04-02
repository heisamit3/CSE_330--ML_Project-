import React from 'react';

function BrandMark({ className = '' }) {
  const classes = ['brand-mark', className].filter(Boolean).join(' ');

  return (
    <svg
      aria-hidden="true"
      className={classes}
      viewBox="0 0 48 48"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        d="M8 24.5L15 32L24 12H40"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="4.5"
      />
      <circle cx="36" cy="10" r="3.75" fill="#f59e0b" />
    </svg>
  );
}

export default BrandMark;

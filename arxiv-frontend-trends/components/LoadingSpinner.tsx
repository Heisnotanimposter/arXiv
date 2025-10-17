
import React from 'react';

const LoadingSpinner: React.FC = () => {
  return (
    <div className="flex justify-center items-center py-20">
      <div className="w-16 h-16 border-4 border-t-4 border-tertiary border-t-accent rounded-full animate-spin"></div>
    </div>
  );
};

export default LoadingSpinner;
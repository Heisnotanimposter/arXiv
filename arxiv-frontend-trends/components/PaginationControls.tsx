import React from 'react';

interface PaginationControlsProps {
  currentPage: number;
  totalPages: number;
  onPageChange: (page: number) => void;
}

const PaginationControls: React.FC<PaginationControlsProps> = ({ currentPage, totalPages, onPageChange }) => {
  const getPageNumbers = () => {
    const pages: (number | string)[] = [];
    const maxPagesToShow = 5;
    const halfPagesToShow = Math.floor(maxPagesToShow / 2);

    if (totalPages <= maxPagesToShow + 2) {
      // Show all pages if total is small
      for (let i = 1; i <= totalPages; i++) {
        pages.push(i);
      }
    } else {
      // Show first page
      pages.push(1);
      
      let startPage = Math.max(2, currentPage - halfPagesToShow);
      let endPage = Math.min(totalPages - 1, currentPage + halfPagesToShow);

      if (currentPage - halfPagesToShow <= 2) {
          endPage = maxPagesToShow;
      }
      if (currentPage + halfPagesToShow >= totalPages - 1) {
          startPage = totalPages - maxPagesToShow + 1;
      }

      // Ellipsis before
      if (startPage > 2) {
        pages.push('...');
      }

      for (let i = startPage; i <= endPage; i++) {
        pages.push(i);
      }
      
      // Ellipsis after
      if (endPage < totalPages - 1) {
        pages.push('...');
      }

      // Show last page
      pages.push(totalPages);
    }
    return pages;
  };

  const pageNumbers = getPageNumbers();

  const commonButtonClasses = "px-4 py-2 text-sm font-semibold rounded-lg transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-primary";
  const activeClasses = "bg-accent text-text-inverted focus:ring-accent";
  const inactiveClasses = "bg-tertiary/50 text-text-base hover:bg-tertiary/80 focus:ring-accent";
  const disabledClasses = "bg-secondary text-text-muted cursor-not-allowed";

  return (
    <nav className="mt-8 flex items-center justify-center space-x-2 animate-fade-in" aria-label="Pagination">
      <button
        onClick={() => onPageChange(currentPage - 1)}
        disabled={currentPage === 1}
        className={`${commonButtonClasses} ${currentPage === 1 ? disabledClasses : inactiveClasses}`}
      >
        Previous
      </button>

      {pageNumbers.map((page, index) =>
        typeof page === 'number' ? (
          <button
            key={index}
            onClick={() => onPageChange(page)}
            className={`${commonButtonClasses} ${currentPage === page ? activeClasses : inactiveClasses}`}
            aria-current={currentPage === page ? 'page' : undefined}
          >
            {page}
          </button>
        ) : (
          <span key={index} className="px-4 py-2 text-sm text-text-muted">
            {page}
          </span>
        )
      )}

      <button
        onClick={() => onPageChange(currentPage + 1)}
        disabled={currentPage === totalPages}
        className={`${commonButtonClasses} ${currentPage === totalPages ? disabledClasses : inactiveClasses}`}
      >
        Next
      </button>
    </nav>
  );
};

export default PaginationControls;
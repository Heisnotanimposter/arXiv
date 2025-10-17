import React from 'react';
import { DateFilter, SortOrder } from '../types';

interface FilterControlsProps {
  dateFilter: DateFilter;
  onDateFilterChange: (filter: DateFilter) => void;
  sortOrder: SortOrder;
  onSortOrderChange: (order: SortOrder) => void;
  articleCount: number;
}

const FilterControls: React.FC<FilterControlsProps> = ({
  dateFilter,
  onDateFilterChange,
  sortOrder,
  onSortOrderChange,
  articleCount,
}) => {
  const commonSelectClasses = "bg-tertiary/50 border border-border-color rounded-md py-2 px-3 text-sm text-text-base focus:outline-none focus:ring-2 focus:ring-accent transition-colors";

  return (
    <div className="my-6 p-4 bg-secondary/50 rounded-lg border border-border-color flex flex-col sm:flex-row items-center justify-between gap-4 animate-fade-in">
      <div className="text-sm text-text-muted">
        <span className="font-bold text-text-base">{articleCount.toLocaleString()}</span> {articleCount === 1 ? 'article' : 'articles'} found
      </div>
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <label htmlFor="dateFilter" className="text-sm font-medium text-text-muted">
            Date:
          </label>
          <select
            id="dateFilter"
            value={dateFilter}
            onChange={(e) => onDateFilterChange(e.target.value as DateFilter)}
            className={commonSelectClasses}
            aria-label="Filter articles by date"
          >
            <option value="all">All Time</option>
            <option value="week">Last Week</option>
            <option value="month">Last Month</option>
            <option value="year">Last Year</option>
          </select>
        </div>
        <div className="flex items-center gap-2">
          <label htmlFor="sortOrder" className="text-sm font-medium text-text-muted">
            Sort by:
          </label>
          <select
            id="sortOrder"
            value={sortOrder}
            onChange={(e) => onSortOrderChange(e.target.value as SortOrder)}
            className={commonSelectClasses}
            aria-label="Sort articles"
          >
            <option value="updated">Latest</option>
            <option value="citations">Citations</option>
            <option value="views">Views</option>
            <option value="impactScore">Impact Score</option>
          </select>
        </div>
      </div>
    </div>
  );
};

export default FilterControls;
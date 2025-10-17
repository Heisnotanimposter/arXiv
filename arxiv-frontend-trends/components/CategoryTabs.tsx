import React from 'react';
import { Category } from '../types';

interface CategoryTabsProps {
  categories: Category[];
  activeCategory: Category | null;
  onSelectCategory: (category: Category) => void;
}

const CategoryTabs: React.FC<CategoryTabsProps> = ({ categories, activeCategory, onSelectCategory }) => {
  return (
    <div className="flex space-x-2 border-b border-border-color">
      {categories.map((category) => (
        <button
          key={category.id}
          onClick={() => onSelectCategory(category)}
          className={`px-4 py-3 text-sm font-medium transition-colors duration-200 focus:outline-none ${
            activeCategory?.id === category.id
              ? 'border-b-2 border-accent text-accent'
              : 'text-text-muted hover:text-text-base'
          }`}
        >
          {category.name}
        </button>
      ))}
    </div>
  );
};

export default CategoryTabs;
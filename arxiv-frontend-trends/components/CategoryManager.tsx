import React, { useState } from 'react';
import { Category } from '../types';

interface CategoryManagerProps {
  initialCategories: Category[];
  onSave: (categories: Category[]) => void;
  onClose: () => void;
}

const CategoryManager: React.FC<CategoryManagerProps> = ({ initialCategories, onSave, onClose }) => {
  const [categories, setCategories] = useState<Category[]>(initialCategories);
  const [newCategoryName, setNewCategoryName] = useState('');
  const [newCategoryQuery, setNewCategoryQuery] = useState('');
  const [error, setError] = useState<string | null>(null);

  const handleAddCategory = () => {
    setError(null);
    if (!newCategoryName.trim() || !newCategoryQuery.trim()) {
      setError('Both name and query are required.');
      return;
    }
    const newCategory: Category = {
      id: `custom-${Date.now()}`,
      name: newCategoryName.trim(),
      query: newCategoryQuery.trim(),
    };
    setCategories([...categories, newCategory]);
    setNewCategoryName('');
    setNewCategoryQuery('');
  };

  const handleDeleteCategory = (id: string) => {
    setCategories(categories.filter(c => c.id !== id));
  };

  const handleSave = () => {
    onSave(categories);
  };

  return (
    <div
      className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 animate-fade-in"
      onClick={onClose}
      aria-modal="true"
      role="dialog"
    >
      <div
        className="bg-secondary rounded-lg border border-border-color shadow-2xl w-full max-w-lg m-4 animate-slide-in-up"
        onClick={e => e.stopPropagation()}
      >
        <div className="p-6 border-b border-border-color">
          <h2 className="text-xl font-bold text-text-base font-serif">Manage Categories</h2>
          <p className="text-sm text-text-muted mt-1">Add or remove categories to customize your feed.</p>
        </div>

        <div className="p-6 max-h-[60vh] overflow-y-auto">
          {categories.length > 0 ? (
            <ul className="space-y-3">
              {categories.map(category => (
                <li key={category.id} className="flex items-center justify-between bg-tertiary/50 p-3 rounded-md">
                  <div>
                    <p className="font-semibold text-text-base">{category.name}</p>
                    <p className="text-xs text-text-muted font-mono">{category.query}</p>
                  </div>
                  <button
                    onClick={() => handleDeleteCategory(category.id)}
                    className="p-2 text-danger-text hover:bg-danger-bg rounded-full transition-colors"
                    aria-label={`Delete category ${category.name}`}
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm4 0a1 1 0 012 0v6a1 1 0 11-2 0V8z" clipRule="evenodd" />
                    </svg>
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-center text-text-muted/80 py-4">You have no categories. Add one below!</p>
          )}
        </div>

        <div className="p-6 border-t border-border-color bg-secondary/50 rounded-b-lg">
          <h3 className="font-semibold text-text-base mb-3">Add New Category</h3>
           {error && (
              <div className="bg-danger-bg text-danger-text p-2 rounded-md mb-3 text-sm" role="alert">
                {error}
              </div>
            )}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label htmlFor="newCategoryName" className="block text-sm font-medium text-text-muted mb-1">
                Category Name
              </label>
              <input
                id="newCategoryName"
                type="text"
                value={newCategoryName}
                onChange={e => setNewCategoryName(e.target.value)}
                placeholder="e.g., React"
                className="w-full bg-tertiary/50 text-text-base placeholder-text-muted/80 rounded-lg py-2 px-3 border border-border-color focus:outline-none focus:ring-2 focus:ring-accent"
              />
            </div>
            <div>
              <div className="flex items-center justify-between mb-1">
                <label htmlFor="newCategoryQuery" className="block text-sm font-medium text-text-muted">
                  arXiv Query
                </label>
                <div className="relative group">
                  <button className="text-text-muted hover:text-accent transition-colors" aria-label="Show query help">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-8-3a1 1 0 00-.867.5 1 1 0 11-1.731-1A3 3 0 0113 8a3.001 3.001 0 01-2 2.83V11a1 1 0 11-2 0v-1a1 1 0 011-1 1 1 0 100-2zm0 8a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
                    </svg>
                  </button>
                  <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-max max-w-xs p-3 bg-primary border border-border-color text-text-muted rounded-lg shadow-lg text-xs opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-opacity duration-300 z-10">
                    <b className="block font-bold text-text-base mb-1">Query Tips:</b>
                    <ul className="list-disc list-inside space-y-1">
                      <li><code className="bg-tertiary p-0.5 rounded text-accent">ti:</code> searches titles.</li>
                      <li><code className="bg-tertiary p-0.5 rounded text-accent">abs:</code> searches abstracts.</li>
                    </ul>
                    <p className="mt-2">Combine with <code className="bg-tertiary p-0.5 rounded">OR</code> / <code className="bg-tertiary p-0.5 rounded">AND</code>.</p>
                    <p className="mt-1 font-mono text-text-muted/80">e.g., <code className="text-accent-2">ti:react AND abs:"virtual dom"</code></p>
                     <div className="absolute left-1/2 -translate-x-1/2 top-full w-0 h-0 border-x-4 border-x-transparent border-t-4 border-t-border-color"></div>
                  </div>
                </div>
              </div>
              <input
                id="newCategoryQuery"
                type="text"
                value={newCategoryQuery}
                onChange={e => setNewCategoryQuery(e.target.value)}
                placeholder="e.g., ti:react"
                className="w-full bg-tertiary/50 text-text-base placeholder-text-muted/80 rounded-lg py-2 px-3 border border-border-color focus:outline-none focus:ring-2 focus:ring-accent"
              />
            </div>
          </div>
          <button
            onClick={handleAddCategory}
            className="w-full mt-4 bg-accent/20 text-accent hover:bg-accent/30 font-semibold py-2 px-4 rounded-lg transition-colors"
          >
            Add Category
          </button>
        </div>

        <div className="p-4 flex justify-end gap-3 bg-primary/50 rounded-b-lg">
          <button
            onClick={onClose}
            className="bg-tertiary/50 text-text-base hover:bg-tertiary/80 font-semibold py-2 px-4 rounded-lg transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            className="bg-accent hover:bg-accent-hover text-text-inverted font-bold py-2 px-4 rounded-lg transition-colors"
          >
            Save Changes
          </button>
        </div>
      </div>
    </div>
  );
};

export default CategoryManager;
import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { Article, Category, DateFilter, SortOrder } from './types';
import { fetchArticles } from './services/arxivService';
import { DEFAULT_CATEGORIES } from './constants';
import Header from './components/Header';
import CategoryTabs from './components/CategoryTabs';
import ArticleCard from './components/ArticleCard';
import LoadingSpinner from './components/LoadingSpinner';
import ErrorMessage from './components/ErrorMessage';
import Login from './components/Login';
import CategoryManager from './components/CategoryManager';
import FilterControls from './components/FilterControls';
import PublicationYearChart from './components/PublicationYearChart';
import PaginationControls from './components/PaginationControls';
import { useTheme } from './hooks/useTheme';

const App: React.FC = () => {
  const [articles, setArticles] = useState<Article[]>([]);
  const [categories, setCategories] = useState<Category[]>([]);
  const [activeCategory, setActiveCategory] = useState<Category | null>(null);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [isLoggedIn, setIsLoggedIn] = useState<boolean>(false);
  const [isCategoryModalOpen, setIsCategoryModalOpen] = useState<boolean>(false);
  const [dateFilter, setDateFilter] = useState<DateFilter>('all');
  const [sortOrder, setSortOrder] = useState<SortOrder>('updated');
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [totalResults, setTotalResults] = useState<number>(0);
  const { theme, setTheme, availableThemes } = useTheme();

  const ARTICLES_PER_PAGE = 15;

  useEffect(() => {
    try {
      const storedCategories = localStorage.getItem('user-categories');
      if (storedCategories) {
        const parsedCategories: Category[] = JSON.parse(storedCategories);
        setCategories(parsedCategories);
        setActiveCategory(parsedCategories[0] || null);
      } else {
        setCategories(DEFAULT_CATEGORIES);
        setActiveCategory(DEFAULT_CATEGORIES[0]);
        if (isLoggedIn) {
          localStorage.setItem('user-categories', JSON.stringify(DEFAULT_CATEGORIES));
        }
      }
    } catch (e) {
      console.error("Failed to parse categories from localStorage", e);
      setCategories(DEFAULT_CATEGORIES);
      setActiveCategory(DEFAULT_CATEGORIES[0]);
       if (isLoggedIn) {
          localStorage.setItem('user-categories', JSON.stringify(DEFAULT_CATEGORIES));
        }
    }
  }, [isLoggedIn]);


  const loadArticles = useCallback(async (query: string, page: number, sort: SortOrder) => {
    setIsLoading(true);
    setError(null);
    setArticles([]);
    try {
      const start = (page - 1) * ARTICLES_PER_PAGE;
      const { articles: fetchedArticles, totalResults: fetchedTotalResults } = await fetchArticles(query, sort, start, ARTICLES_PER_PAGE);
      setArticles(fetchedArticles);
      setTotalResults(fetchedTotalResults);
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('An unknown error occurred.');
      }
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!isLoggedIn) return; 

    let queryToFetch = '';
    if (searchQuery) {
      queryToFetch = `all:"${searchQuery}"`;
    } else if (activeCategory) {
      queryToFetch = activeCategory.query;
    }

    if (queryToFetch) {
      loadArticles(queryToFetch, currentPage, sortOrder);
    } else {
      setIsLoading(false);
      setArticles([]);
      setTotalResults(0);
    }
  }, [activeCategory, searchQuery, loadArticles, isLoggedIn, currentPage, sortOrder]);

  const filteredArticles = useMemo(() => {
    let processedArticles = [...articles];

    // Filter by date
    if (dateFilter !== 'all') {
      const now = new Date();
      processedArticles = processedArticles.filter(article => {
        const articleDate = new Date(article.published);
        const diffTime = now.getTime() - articleDate.getTime();
        const diffDays = diffTime / (1000 * 60 * 60 * 24);

        switch (dateFilter) {
          case 'week':
            return diffDays <= 7;
          case 'month':
            return diffDays <= 30;
          case 'year':
            return diffDays <= 365;
          default:
            return true;
        }
      });
    }
    
    return processedArticles;
  }, [articles, dateFilter]);


  const handleSearch = (query: string) => {
    const trimmedQuery = query.trim();
    if (trimmedQuery === searchQuery) return;
    
    setCurrentPage(1); 
    setSearchQuery(trimmedQuery);
    if (trimmedQuery) {
      setActiveCategory(null);
    } else if (!activeCategory && categories.length > 0) {
      setActiveCategory(categories[0]);
    }
  };

  const handleSelectCategory = (category: Category) => {
    setCurrentPage(1); 
    setSearchQuery('');
    setActiveCategory(category);
  };
  
  const handleSortOrderChange = (order: SortOrder) => {
    setSortOrder(order);
    setCurrentPage(1);
  };

  const handlePageChange = (page: number) => {
    setCurrentPage(page);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const totalPages = useMemo(() => Math.ceil(totalResults / ARTICLES_PER_PAGE), [totalResults]);

  const handleLoginSuccess = () => {
    setIsLoggedIn(true);
  };

  const handleLogout = () => {
    setIsLoggedIn(false);
    setArticles([]);
    if (categories.length > 0) {
      setActiveCategory(categories[0]);
    }
    setSearchQuery('');
    setError(null);
    setIsLoading(true);
    setCurrentPage(1);
    setTotalResults(0);
  };

  const handleSaveCategories = (newCategories: Category[]) => {
    setCategories(newCategories);
    localStorage.setItem('user-categories', JSON.stringify(newCategories));
    
    if (activeCategory && !newCategories.some(c => c.id === activeCategory.id)) {
      setActiveCategory(newCategories[0] || null);
    } else if (!activeCategory && newCategories.length > 0) {
      setActiveCategory(newCategories[0] || null);
    }
    
    setIsCategoryModalOpen(false);
  };

  if (!isLoggedIn) {
    return <Login onLoginSuccess={handleLoginSuccess} />;
  }

  return (
    <div className="min-h-screen bg-primary text-text-base animate-fade-in">
      <Header
        onSearch={handleSearch}
        isLoggedIn={isLoggedIn}
        onLogout={handleLogout}
        onManageCategories={() => setIsCategoryModalOpen(true)}
        theme={theme}
        setTheme={setTheme}
        availableThemes={availableThemes}
      />
      <main className="container mx-auto px-4 py-8 max-w-5xl">
        <CategoryTabs
          categories={categories}
          activeCategory={activeCategory}
          onSelectCategory={handleSelectCategory}
        />
        
        <FilterControls
            dateFilter={dateFilter}
            onDateFilterChange={setDateFilter}
            sortOrder={sortOrder}
            onSortOrderChange={handleSortOrderChange}
            articleCount={totalResults}
        />
        
        <PublicationYearChart articles={filteredArticles} />

        <div className="mt-2">
          {isLoading && <LoadingSpinner />}
          {error && <ErrorMessage message={error} />}
          {!isLoading && !error && (
            <div className="grid grid-cols-1 gap-6">
              {filteredArticles.length > 0 ? (
                filteredArticles.map((article, index) => (
                  <ArticleCard key={article.id} article={article} index={index} />
                ))
              ) : (
                 <div className="text-center py-12 bg-secondary rounded-lg">
                  <h3 className="text-xl text-text-muted">
                    {articles.length > 0 ? 'No articles match your current filters.' : (searchQuery ? `No articles found for "${searchQuery}"` : (activeCategory ? `No articles found for "${activeCategory.name}"` : 'No category selected'))}
                  </h3>
                   <p className="text-text-muted/80 mt-2">
                     {articles.length > 0 ? 'Try adjusting the date range or sort order.' : (categories.length === 0 ? "Try adding a category in the settings." : "Try a different search term or category.")}
                   </p>
                </div>
              )}
            </div>
          )}
        </div>
        
        {!isLoading && !error && totalPages > 1 && (
          <PaginationControls
            currentPage={currentPage}
            totalPages={totalPages}
            onPageChange={handlePageChange}
          />
        )}

      </main>
      {isCategoryModalOpen && (
        <CategoryManager
          initialCategories={categories}
          onSave={handleSaveCategories}
          onClose={() => setIsCategoryModalOpen(false)}
        />
      )}
    </div>
  );
};

export default App;
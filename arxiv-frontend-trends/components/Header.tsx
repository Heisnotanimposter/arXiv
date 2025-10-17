import React, { useState } from 'react';
import { Theme, ThemeName } from '../types';

interface HeaderProps {
  onSearch: (query: string) => void;
  isLoggedIn: boolean;
  onLogout: () => void;
  onManageCategories: () => void;
  theme: Theme | undefined;
  setTheme: (themeId: ThemeName) => void;
  availableThemes: Theme[];
}

const ThemeSwitcher: React.FC<Pick<HeaderProps, 'theme' | 'setTheme' | 'availableThemes'>> = ({ theme, setTheme, availableThemes }) => {
    const [isOpen, setIsOpen] = useState(false);

    return (
        <div className="relative">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="flex-shrink-0 bg-tertiary/50 text-text-muted hover:bg-tertiary/80 font-semibold p-2.5 rounded-full transition-colors"
                aria-label="Change theme"
            >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                  <path d="M10 2a1 1 0 00-1 1v1a1 1 0 002 0V3a1 1 0 00-1-1zM4 10a1 1 0 01-1-1H2a1 1 0 010-2h1a1 1 0 011 1zM16 9a1 1 0 011 1h1a1 1 0 110 2h-1a1 1 0 01-1-1zM10 16a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM4.22 5.64a1 1 0 011.414 0l.707.707a1 1 0 01-1.414 1.414l-.707-.707a1 1 0 010-1.414zM14.36 14.36a1 1 0 011.414 0l.707.707a1 1 0 01-1.414 1.414l-.707-.707a1 1 0 010-1.414zM15.78 5.64a1 1 0 010 1.414l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 0zM5.64 14.36a1 1 0 010 1.414l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 0zM10 5a5 5 0 100 10 5 5 0 000-10z" />
                </svg>
            </button>
            {isOpen && (
                <div 
                    className="absolute right-0 mt-2 w-40 bg-secondary border border-border-color rounded-lg shadow-lg z-20 animate-fade-in"
                    onMouseLeave={() => setIsOpen(false)}
                >
                    <ul className="py-1">
                        {availableThemes.map(t => (
                            <li key={t.id}>
                                <button
                                    onClick={() => {
                                        setTheme(t.id);
                                        setIsOpen(false);
                                    }}
                                    className={`w-full text-left px-4 py-2 text-sm ${
                                        theme?.id === t.id
                                            ? 'bg-accent/30 text-accent'
                                            : 'text-text-base hover:bg-tertiary/50'
                                    }`}
                                >
                                    {t.name}
                                </button>
                            </li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
};

const Header: React.FC<HeaderProps> = ({ onSearch, isLoggedIn, onLogout, onManageCategories, theme, setTheme, availableThemes }) => {
  const [query, setQuery] = useState('');

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    onSearch(query);
  };

  return (
    <header className="bg-secondary/50 backdrop-blur-sm sticky top-0 z-10 border-b border-border-color">
      <div className="container mx-auto px-4 py-4 max-w-5xl flex flex-col sm:flex-row justify-between sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-accent to-accent-2">
            ArXiv Frontend Trends
          </h1>
          <p className="text-text-muted mt-1 font-sans">
            Discover the latest research in frontend technologies.
          </p>
        </div>
        <div className="flex items-center gap-2 sm:gap-4">
          <form onSubmit={handleSubmit} className="flex-grow sm:flex-grow-0">
            <div className="relative">
              <input
                type="search"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search articles..."
                className="w-full sm:w-64 bg-tertiary/50 text-text-base placeholder-text-muted rounded-full py-2 pl-4 pr-10 border border-transparent focus:outline-none focus:ring-2 focus:ring-accent-hover focus:border-transparent transition-all"
                aria-label="Search articles"
              />
              <button
                type="submit"
                className="absolute inset-y-0 right-0 flex items-center justify-center px-3 text-text-muted hover:text-accent transition-colors"
                aria-label="Submit search"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M8 4a4 4 0 100 8 4 4 0 000-8zM2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8z" clipRule="evenodd" />
                </svg>
              </button>
            </div>
          </form>
          {isLoggedIn && (
            <>
              <ThemeSwitcher theme={theme} setTheme={setTheme} availableThemes={availableThemes} />
              <button
                onClick={onManageCategories}
                className="flex-shrink-0 bg-tertiary/50 text-text-muted hover:bg-tertiary/80 font-semibold p-2.5 rounded-full transition-colors"
                aria-label="Manage Categories"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M11.49 3.17c-.38-1.56-2.6-1.56-2.98 0L8.12 5.12c-.67.21-1.31.52-1.9.91L4.35 4.16c-1.28-.73-2.82.21-2.5 1.63l.34 1.48c.18.75.52 1.44.99 2.05l-1.21 1.21c-1.03 1.03-.38 2.82 1.14 3.14l1.62.34c.64.13 1.25.41 1.81.78l.47 1.87c.32 1.28 2.04 1.63 3.16.94l1.58-.98c.57-.35 1.2-.6 1.84-.71l1.76-.23c1.55-.2 2.14-1.96.96-3.05l-1.2-1.2c-.47-.47-.81-1.03-1.02-1.64l-.34-1.48c-.22-1.42-1.76-2.36-3.04-1.63l-1.87 1.07c-.59.34-1.25.59-1.94.7L11.49 3.17zM10 13a3 3 0 100-6 3 3 0 000 6z" clipRule="evenodd" />
                </svg>
              </button>
              <button
                onClick={onLogout}
                className="flex-shrink-0 bg-danger-bg border border-danger-border text-danger-text hover:bg-danger-bg/80 font-semibold py-2 px-4 rounded-full transition-colors"
                aria-label="Logout"
              >
                Logout
              </button>
            </>
          )}
        </div>
      </div>
    </header>
  );
};

export default Header;
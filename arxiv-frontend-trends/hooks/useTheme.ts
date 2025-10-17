import { useState, useEffect, useCallback } from 'react';
import { Theme, ThemeName, ThemeProperties } from '../types';
import { THEMES } from '../themes';

const applyTheme = (properties: ThemeProperties) => {
  const root = document.documentElement;
  Object.entries(properties).forEach(([property, value]) => {
    root.style.setProperty(property, value);
  });
};

export const useTheme = () => {
  const [themeId, setThemeId] = useState<ThemeName>(() => {
    try {
      const storedTheme = window.localStorage.getItem('app-theme') as ThemeName;
      return storedTheme && THEMES.some(t => t.id === storedTheme) ? storedTheme : 'slate';
    } catch (error) {
      return 'slate';
    }
  });

  useEffect(() => {
    const currentTheme = THEMES.find(t => t.id === themeId);
    if (currentTheme) {
      applyTheme(currentTheme.properties);
      try {
        window.localStorage.setItem('app-theme', themeId);
      } catch (error) {
        console.error("Failed to save theme to localStorage", error);
      }
    }
  }, [themeId]);

  const setTheme = useCallback((newThemeId: ThemeName) => {
    setThemeId(newThemeId);
  }, []);
  
  const currentTheme = THEMES.find(t => t.id === themeId);

  return {
    theme: currentTheme,
    setTheme,
    availableThemes: THEMES
  };
};
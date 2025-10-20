export interface Article {
  id: string;
  title: string;
  summary: string;
  authors: string[];
  published: string;
  updated: string;
  link: string;
  citations: number;
  views: number;
  impactScore: number;
}

export interface Category {
  id: string;
  name: string;
  query: string;
}

export type DateFilter = 'all' | 'week' | 'month' | 'year';
export type SortOrder = 'updated' | 'citations' | 'views' | 'impactScore';

export interface KeywordAnalysis {
  keyword: string;
  frequency: number;
}

export type ThemeName = 'slate' | 'cyberpunk' | 'parchment' | 'forest';

export interface ThemeProperties {
  '--color-primary': string;
  '--color-secondary': string;
  '--color-tertiary': string;
  '--color-accent': string;
  '--color-accent-hover': string;
  '--color-accent-2': string;
  '--color-accent-3': string;
  '--color-text-base': string;
  '--color-text-muted': string;
  '--color-text-inverted': string;
  '--color-border-color': string;
  '--color-danger-bg': string;
  '--color-danger-border': string;
  '--color-danger-text': string;
  '--font-sans': string;
  '--font-serif': string;
  '--font-mono': string;
}

export interface Theme {
  id: ThemeName;
  name: string;
  properties: ThemeProperties;
}
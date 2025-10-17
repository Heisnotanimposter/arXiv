import { Category } from './types';

export const DEFAULT_CATEGORIES: Category[] = [
  { id: 'javascript', name: 'JavaScript', query: 'ti:javascript OR abs:javascript' },
  { id: 'css', name: 'CSS', query: 'ti:"cascading style sheets" OR abs:"cascading style sheets"' },
  { id: 'html', name: 'HTML', query: 'ti:"hypertext markup language" OR abs:"hypertext markup language"' },
];

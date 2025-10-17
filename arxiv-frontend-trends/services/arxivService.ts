import { Article, SortOrder } from '../types';

const ARXIV_API_BASE_URL = 'https://export.arxiv.org/api/query?';

// Helper to safely get text content from an XML element
const getElementText = (element: Element, tagName: string): string => {
  const node = element.querySelector(tagName);
  return node?.textContent || 'N/A';
};

// Helper function for simulating metrics
const generateSimulatedMetrics = (publishedDate: string): { citations: number; views: number; impactScore: number } => {
  const now = new Date();
  const pubDate = new Date(publishedDate);
  const daysSincePublished = Math.max(1, Math.floor((now.getTime() - pubDate.getTime()) / (1000 * 3600 * 24)));

  // Simulate citations: older papers have a higher chance of more citations
  const baseCitations = Math.random() * (daysSincePublished / 50);
  const citations = Math.floor(baseCitations * (Math.random() * 5 + 1));

  // Simulate views: generally much higher than citations
  const views = citations * (Math.floor(Math.random() * 20) + 10) + Math.floor(Math.random() * 500);

  // Simulate impact score: citations per year (normalized)
  const yearsSincePublished = daysSincePublished / 365.25;
  const impactScore = yearsSincePublished > 0 ? parseFloat((citations / yearsSincePublished).toFixed(2)) : citations * 2;
  
  return {
    citations,
    views,
    impactScore,
  };
};

export const fetchArticles = async (
  query: string,
  sortOrder: SortOrder,
  start: number = 0,
  maxResults: number = 15
): Promise<{ articles: Article[]; totalResults: number }> => {
  const sortBy = sortOrder === 'updated' ? 'lastUpdatedDate' : 'relevance';
  
  const params = new URLSearchParams({
    search_query: query,
    sortBy: sortBy,
    sortOrder: 'descending',
    start: start.toString(),
    max_results: maxResults.toString(),
  });

  const url = `${ARXIV_API_BASE_URL}${params.toString()}`;

  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`arXiv API request failed with status ${response.status}`);
    }

    const xmlText = await response.text();
    const parser = new DOMParser();
    const xmlDoc = parser.parseFromString(xmlText, 'application/xml');
    
    const errorNode = xmlDoc.querySelector('entry title');
    if (errorNode && errorNode.textContent?.includes('Error')) {
      const summary = getElementText(xmlDoc.querySelector('entry')!, 'summary');
      throw new Error(`arXiv API Error: ${summary}`);
    }
    
    const totalResultsNode = xmlDoc.querySelector('totalResults');
    const totalResults = totalResultsNode ? parseInt(totalResultsNode.textContent || '0', 10) : 0;

    const entries = Array.from(xmlDoc.querySelectorAll('entry'));
    if (entries.length === 0) {
      return { articles: [], totalResults };
    }

    const articles = entries.map((entry): Article => {
      const authors = Array.from(entry.querySelectorAll('author name')).map(
        (authorNode) => authorNode.textContent || 'Unknown Author'
      );
      
      const linkNode = entry.querySelector('link[rel="alternate"][type="text/html"]');
      const published = getElementText(entry, 'published');
      const { citations, views, impactScore } = generateSimulatedMetrics(published);
      
      return {
        id: getElementText(entry, 'id'),
        title: getElementText(entry, 'title').replace(/\s+/g, ' ').trim(),
        summary: getElementText(entry, 'summary').replace(/\s+/g, ' ').trim(),
        authors,
        published,
        updated: getElementText(entry, 'updated'),
        link: linkNode?.getAttribute('href') || '#',
        citations,
        views,
        impactScore,
      };
    });
    
    return { articles, totalResults };

  } catch (error) {
    console.error('Failed to fetch or parse arXiv articles:', error);
    throw new Error('Could not fetch articles from arXiv. Please check your network connection or try again later.');
  }
};
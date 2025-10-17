import React, { useMemo } from 'react';
import { Article } from '../types';

interface PublicationYearChartProps {
  articles: Article[];
}

const PublicationYearChart: React.FC<PublicationYearChartProps> = ({ articles }) => {
  const chartData = useMemo(() => {
    const currentYear = new Date().getFullYear();
    const years = Array.from({ length: 5 }, (_, i) => currentYear - i).reverse();

    const yearCounts: { [year: number]: number } = years.reduce((acc, year) => {
      acc[year] = 0;
      return acc;
    }, {} as { [year: number]: number });

    articles.forEach(article => {
      const publishedYear = new Date(article.published).getFullYear();
      if (yearCounts.hasOwnProperty(publishedYear)) {
        yearCounts[publishedYear]++;
      }
    });

    const dataPoints = years.map(year => ({
      year,
      count: yearCounts[year],
    }));

    const maxCount = Math.max(...dataPoints.map(d => d.count), 1); // Avoid division by zero

    return { dataPoints, maxCount };
  }, [articles]);

  const { dataPoints, maxCount } = chartData;
  const totalArticles = dataPoints.reduce((sum, item) => sum + item.count, 0);

  // Do not render the chart if there's no data for the past 5 years.
  if (totalArticles === 0) {
    return null;
  }
  
  return (
    <div className="my-6 p-6 bg-secondary/50 rounded-lg border border-border-color animate-fade-in">
      <h3 className="text-lg font-semibold text-text-base mb-4 font-serif">
        Publications Over Last 5 Years
      </h3>
      <div className="flex justify-around items-end h-48 pt-4 border-b border-l border-border-color/50">
        {dataPoints.map(({ year, count }) => {
          const barHeight = count === 0 ? '0%' : `${Math.max((count / maxCount) * 100, 2)}%`; // give a minimum height for visibility
          return (
            <div key={year} className="flex flex-col items-center h-full w-1/6" title={`${count} articles in ${year}`}>
              <div className="text-sm font-bold text-accent mb-1">{count}</div>
              <div className="w-1/2 h-full flex items-end group">
                <div
                  className="w-full bg-accent/80 group-hover:bg-accent rounded-t-sm transition-all duration-300 ease-in-out"
                  style={{ height: barHeight }}
                ></div>
              </div>
              <div className="text-xs font-medium text-text-muted mt-2 border-t-2 border-border-color w-full text-center pt-1">{year}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default PublicationYearChart;

import React, { useState } from 'react';
import { Article, KeywordAnalysis } from '../types';
import { analyzeArticleKeywords } from '../services/geminiService';
import KeywordAnalysisModal from './KeywordAnalysisModal';


interface ArticleCardProps {
  article: Article;
  index: number;
}

const SUMMARY_MAX_LENGTH = 280;

const MetricBar: React.FC<{ value: number; maxValue: number; colorClass: string }> = ({ value, maxValue, colorClass }) => {
  const percentage = Math.min(100, (value / maxValue) * 100);
  return (
    <div className="w-full bg-tertiary/70 rounded-full h-2 overflow-hidden">
      <div
        className={`${colorClass} h-2 rounded-full transition-all duration-1000 ease-out`}
        style={{ width: `${percentage}%` }}
      ></div>
    </div>
  );
};

const ArticleCard: React.FC<ArticleCardProps> = ({ article, index }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [analysisData, setAnalysisData] = useState<KeywordAnalysis[] | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    });
  };

  const isLongSummary = article.summary.length > SUMMARY_MAX_LENGTH;

  const summaryText = isLongSummary && !isExpanded
    ? `${article.summary.substring(0, SUMMARY_MAX_LENGTH)}...`
    : article.summary;

  const handleToggleSummary = () => {
    setIsExpanded(!isExpanded);
  };
  
  const handleAnalyzeClick = async () => {
    setIsModalOpen(true);
    // Don't re-fetch if data already exists
    if(analysisData) return;

    setIsAnalyzing(true);
    setAnalysisError(null);
    setAnalysisData(null);

    try {
      const results = await analyzeArticleKeywords(article.title, article.summary);
      setAnalysisData(results);
    } catch (err) {
      if (err instanceof Error) {
        setAnalysisError(err.message);
      } else {
        setAnalysisError('An unknown error occurred during analysis.');
      }
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleCloseModal = () => {
    setIsModalOpen(false);
  };


  return (
    <>
      <div
        className="bg-secondary p-6 rounded-lg border border-border-color hover:border-accent transition-all duration-300 transform hover:-translate-y-1 animate-slide-in-up"
        style={{ animationDelay: `${index * 50}ms`, animationFillMode: 'both' }}
      >
        <div className="flex flex-col md:flex-row justify-between md:items-start mb-2">
          <h2 className="text-xl font-bold text-text-base mb-2 md:mb-0 pr-4 font-serif">
            {article.title}
          </h2>
          <a
            href={article.link}
            target="_blank"
            rel="noopener noreferrer"
            className="flex-shrink-0 inline-flex items-center px-4 py-2 text-sm font-semibold text-accent bg-accent/10 rounded-full hover:bg-accent/20 transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-accent focus:ring-offset-2 focus:ring-offset-secondary"
          >
            Read on ArXiv
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 ml-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
            </svg>
          </a>
        </div>
        <div className="text-xs text-text-muted mb-4 font-sans">
          <p>
            <span className="font-semibold">Authors:</span> {article.authors.join(', ')}
          </p>
          <p>
            <span className="font-semibold">Last Updated:</span> {formatDate(article.updated)}
          </p>
        </div>
        <div className="text-text-base/90 leading-relaxed text-sm mb-6 font-sans">
          <p>
            {summaryText}
            {isLongSummary && (
              <button
                onClick={handleToggleSummary}
                className="text-accent hover:text-accent-hover font-semibold ml-2 focus:outline-none transition-colors"
                aria-expanded={isExpanded}
              >
                {isExpanded ? 'Show Less' : 'Read More'}
              </button>
            )}
          </p>
        </div>
        
        <div className="border-t border-border-color pt-4">
          <h3 className="text-sm font-semibold text-text-muted mb-3 tracking-wider uppercase font-sans">Trend Metrics</h3>
          <div className="space-y-3">
            <div className="grid grid-cols-12 items-center gap-x-4">
              <span className="text-sm font-medium text-text-base col-span-3">Citations</span>
              <div className="col-span-7">
                <MetricBar value={article.citations} maxValue={500} colorClass="bg-accent" />
              </div>
              <span className="text-sm font-bold text-accent col-span-2 text-right">{article.citations.toLocaleString()}</span>
            </div>
            <div className="grid grid-cols-12 items-center gap-x-4">
              <span className="text-sm font-medium text-text-base col-span-3">Views</span>
              <div className="col-span-7">
                <MetricBar value={article.views} maxValue={20000} colorClass="bg-accent-2" />
              </div>
              <span className="text-sm font-bold text-accent-2 col-span-2 text-right">{article.views.toLocaleString()}</span>
            </div>
            <div className="grid grid-cols-12 items-center gap-x-4">
              <span className="text-sm font-medium text-text-base col-span-3">Impact Score</span>
              <div className="col-span-7">
                <MetricBar value={article.impactScore} maxValue={100} colorClass="bg-accent-3" />
              </div>
              <span className="text-sm font-bold text-accent-3 col-span-2 text-right">{article.impactScore.toLocaleString()}</span>
            </div>
          </div>
          <div className="mt-5">
            <button
              onClick={handleAnalyzeClick}
              disabled={isAnalyzing}
              className="w-full bg-tertiary/60 text-text-base hover:bg-tertiary/90 font-semibold py-2 px-4 rounded-lg transition-colors flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-wait focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-secondary focus:ring-accent"
            >
              {isAnalyzing ? (
                <>
                  <div className="w-4 h-4 border-2 border-t-2 border-text-muted border-t-accent rounded-full animate-spin"></div>
                  <span>Analyzing...</span>
                </>
              ) : (
                <>
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                    <path d="M10 3.5a1.5 1.5 0 011.5 1.5v1.5a1.5 1.5 0 01-3 0V5A1.5 1.5 0 0110 3.5z" />
                    <path fillRule="evenodd" d="M3.75 7.5a1.5 1.5 0 011.5-1.5h9.5a1.5 1.5 0 011.5 1.5v5.5a1.5 1.5 0 01-1.5 1.5h-2.5a.75.75 0 000 1.5h2.5a3 3 0 003-3V7.5a3 3 0 00-3-3h-9.5a3 3 0 00-3 3v5.5a3 3 0 003 3h2.5a.75.75 0 000-1.5h-2.5a1.5 1.5 0 01-1.5-1.5V7.5z" clipRule="evenodd" />
                  </svg>
                  <span>Analyze Keywords</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>
      <KeywordAnalysisModal
        isOpen={isModalOpen}
        onClose={handleCloseModal}
        isLoading={isAnalyzing}
        error={analysisError}
        data={analysisData}
        articleTitle={article.title}
      />
    </>
  );
};

export default ArticleCard;
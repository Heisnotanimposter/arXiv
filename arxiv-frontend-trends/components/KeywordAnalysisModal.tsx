import React from 'react';
import { KeywordAnalysis } from '../types';
import LoadingSpinner from './LoadingSpinner';
import ErrorMessage from './ErrorMessage';

interface KeywordAnalysisModalProps {
  isOpen: boolean;
  onClose: () => void;
  isLoading: boolean;
  error: string | null;
  data: KeywordAnalysis[] | null;
  articleTitle: string;
}

const KeywordAnalysisModal: React.FC<KeywordAnalysisModalProps> = ({
  isOpen,
  onClose,
  isLoading,
  error,
  data,
  articleTitle
}) => {
  if (!isOpen) return null;

  const maxFrequency = data ? Math.max(...data.map(d => d.frequency), 1) : 1;

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
          <h2 className="text-xl font-bold text-text-base font-serif">Keyword Analysis</h2>
          <p className="text-sm text-text-muted mt-1 truncate">
            For: <span className="italic">{articleTitle}</span>
          </p>
        </div>

        <div className="p-6 max-h-[60vh] overflow-y-auto">
          {isLoading && <LoadingSpinner />}
          {error && <ErrorMessage message={error} />}
          {data && (
             <div className="flow-root">
                <div className="-mx-6">
                <div className="inline-block min-w-full align-middle">
                    <table className="min-w-full divide-y divide-border-color">
                    <thead>
                        <tr>
                        <th scope="col" className="py-3.5 px-6 text-left text-sm font-semibold text-text-base">
                            Keyword
                        </th>
                        <th scope="col" className="py-3.5 px-6 text-left text-sm font-semibold text-text-base">
                            Frequency
                        </th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-border-color/50">
                        {data.map(({ keyword, frequency }) => (
                        <tr key={keyword}>
                            <td className="whitespace-nowrap py-4 px-6 text-sm font-medium text-text-base">
                                {keyword}
                            </td>
                            <td className="whitespace-nowrap py-4 px-6 text-sm text-text-muted">
                                <div className="flex items-center gap-4">
                                    <div className="w-full bg-tertiary rounded-full h-2.5">
                                        <div 
                                            className="bg-accent h-2.5 rounded-full" 
                                            style={{width: `${(frequency / maxFrequency) * 100}%`}}>
                                        </div>
                                    </div>
                                    <span className="font-bold text-accent w-4 text-right">{frequency}</span>
                                </div>
                            </td>
                        </tr>
                        ))}
                    </tbody>
                    </table>
                </div>
                </div>
            </div>
          )}
           {data?.length === 0 && !isLoading && !error && (
            <div className="text-center text-text-muted/80 py-8">
              <p>No significant keywords were identified in this article.</p>
            </div>
          )}
        </div>
        
        <div className="p-4 flex justify-end gap-3 bg-primary/50 rounded-b-lg border-t border-border-color">
          <button
            onClick={onClose}
            className="bg-accent hover:bg-accent-hover text-text-inverted font-bold py-2 px-4 rounded-lg transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

export default KeywordAnalysisModal;
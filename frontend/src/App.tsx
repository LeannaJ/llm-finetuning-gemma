import React, { useState } from 'react';
import Header from './components/Header.tsx';
import ImageUpload from './components/ImageUpload.tsx';
import AnalysisResult from './components/AnalysisResult.tsx';
import { AnalysisData } from './types.ts';

function App() {
  const [analysisData, setAnalysisData] = useState<AnalysisData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleAnalysisComplete = (data: AnalysisData) => {
    setAnalysisData(data);
    setIsLoading(false);
    setError(null);
  };

  const handleAnalysisError = (errorMessage: string) => {
    setError(errorMessage);
    setIsLoading(false);
    setAnalysisData(null);
  };

  const handleAnalysisStart = () => {
    setIsLoading(true);
    setError(null);
    setAnalysisData(null);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      
      <main className="container mx-auto px-4 py-8 max-w-4xl">
        <div className="space-y-8">
          {/* Main upload section */}
          <ImageUpload 
            onAnalysisStart={handleAnalysisStart}
            onAnalysisComplete={handleAnalysisComplete}
            onAnalysisError={handleAnalysisError}
            isLoading={isLoading}
          />
          
          {/* Loading State */}
          {isLoading && (
            <div className="card text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto mb-4"></div>
              <p className="text-gray-600">Evaluating content...</p>
              <p className="text-sm text-gray-500 mt-2">Please wait a moment</p>
            </div>
          )}
          
          {/* Error Message */}
          {error && (
            <div className="card border-red-200 bg-red-50">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                  </svg>
                </div>
                <div className="ml-3">
                  <h3 className="text-sm font-medium text-red-800">Error occurred during evaluation</h3>
                  <p className="text-sm text-red-700 mt-1">{error}</p>
                </div>
              </div>
            </div>
          )}
          
          {/* Evaluation result */}
          {analysisData && (
            <AnalysisResult data={analysisData} />
          )}
        </div>
      </main>
      
      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-16">
        <div className="container mx-auto px-4 py-8 text-center text-gray-600">
          <p>&copy; 2025 ThinkGrade. AI-powered Educational Assessment</p>
        </div>
      </footer>
    </div>
  );
}

export default App; 
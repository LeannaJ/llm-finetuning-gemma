import React from 'react';
import { CheckCircle, Star, Lightbulb, TrendingUp } from 'lucide-react';
import { AnalysisData } from '../types.ts';

interface AnalysisResultProps {
  data: AnalysisData;
}

const AnalysisResult: React.FC<AnalysisResultProps> = ({ data }) => {
  const { extracted_text, feedback, subject } = data;

  const getSubjectLabel = (subject: string) => {
    switch (subject) {
      case 'math': return 'Math Problem';
      case 'essay': return 'Essay/Summary';
      case 'notes': return 'Study Notes';
      default: return subject;
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 90) return 'text-green-600';
    if (score >= 80) return 'text-blue-600';
    if (score >= 70) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getScoreIcon = (score: number) => {
    if (score >= 90) return '🏆';
    if (score >= 80) return '⭐';
    if (score >= 70) return '👍';
    return '📝';
  };

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Result Header */}
      <div className="card bg-gradient-to-r from-primary-50 to-blue-50 border-primary-200">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-gray-900 mb-2">
              Evaluation Result
            </h2>
            <p className="text-gray-600">
              {getSubjectLabel(subject)} evaluation completed
            </p>
          </div>
          <div className="text-right">
            <div className={`text-3xl font-bold ${getScoreColor(feedback.score)}`}>
              {feedback.score}/100
            </div>
            <div className="text-2xl">{getScoreIcon(feedback.score)}</div>
          </div>
        </div>
      </div>

      {/* Extracted Text */}
      <div className="card">
        <div className="flex items-center mb-4">
          <CheckCircle className="h-5 w-5 text-green-500 mr-2" />
          <h3 className="text-lg font-semibold text-gray-900">Extracted Text</h3>
        </div>
        <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
          <p className="text-gray-800 whitespace-pre-wrap font-mono">
            {extracted_text || 'Unable to extract text.'}
          </p>
        </div>
      </div>

      {/* Analysis Results */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Evaluation Content */}
        <div className="card">
          <div className="flex items-center mb-4">
            <TrendingUp className="h-5 w-5 text-blue-500 mr-2" />
            <h3 className="text-lg font-semibold text-gray-900">Evaluation Details</h3>
          </div>
          <div className="space-y-3">
            {feedback.analysis ? (
              <p className="text-gray-700 leading-relaxed">
                {feedback.analysis}
              </p>
            ) : (
              <p className="text-gray-500 italic">
                No evaluation details available.
              </p>
            )}
          </div>
        </div>

        {/* Improvement Suggestions */}
        <div className="card">
          <div className="flex items-center mb-4">
            <Lightbulb className="h-5 w-5 text-yellow-500 mr-2" />
            <h3 className="text-lg font-semibold text-gray-900">Learning Tips</h3>
          </div>
          <div className="space-y-3">
            {feedback.suggestions && feedback.suggestions.length > 0 ? (
              <ul className="space-y-2">
                {feedback.suggestions.map((suggestion, index) => (
                  <li key={index} className="flex items-start">
                    <span className="text-primary-500 mr-2 mt-1">•</span>
                    <span className="text-gray-700">{suggestion}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-gray-500 italic">
                No learning tips available.
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Overall Summary */}
      <div className="card bg-gradient-to-r from-gray-50 to-gray-100">
        <div className="flex items-center mb-4">
          <Star className="h-5 w-5 text-yellow-500 mr-2" />
          <h3 className="text-lg font-semibold text-gray-900">Overall Assessment</h3>
        </div>
        <div className="bg-white rounded-lg p-4 border border-gray-200">
          <p className="text-gray-700 leading-relaxed">
            {feedback.summary || 'Unable to generate assessment.'}
          </p>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex justify-center space-x-4">
        <button
          onClick={() => window.location.reload()}
          className="btn-primary"
        >
          Start New Evaluation
        </button>
        <button
          onClick={() => {
            const text = `Evaluation Result\n\nSubject: ${getSubjectLabel(subject)}\nScore: ${feedback.score}/100\n\nExtracted Text:\n${extracted_text}\n\nEvaluation Details:\n${feedback.analysis}\n\nLearning Tips:\n${feedback.suggestions.join('\n')}\n\nOverall Assessment:\n${feedback.summary}`;
            navigator.clipboard.writeText(text);
            alert('Result copied to clipboard!');
          }}
          className="btn-secondary"
        >
          Copy Result
        </button>
      </div>
    </div>
  );
};

export default AnalysisResult; 
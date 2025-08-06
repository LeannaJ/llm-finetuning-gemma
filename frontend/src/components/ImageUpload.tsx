import React, { useState, useRef } from 'react';
import axios from 'axios';
import { Upload, Camera, FileText, Calculator } from 'lucide-react';
import { AnalysisData, SubjectType } from '../types.ts';

interface ImageUploadProps {
  onAnalysisStart: () => void;
  onAnalysisComplete: (data: AnalysisData) => void;
  onAnalysisError: (error: string) => void;
  isLoading: boolean;
}

const ImageUpload: React.FC<ImageUploadProps> = ({
  onAnalysisStart,
  onAnalysisComplete,
  onAnalysisError,
  isLoading
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [subject, setSubject] = useState<SubjectType>('math');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    } else {
      onAnalysisError('Please upload an image file only.');
    }
  };

  const handleDragOver = (event: React.DragEvent) => {
    event.preventDefault();
  };

  const handleDrop = (event: React.DragEvent) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    } else {
      onAnalysisError('Please upload an image file only.');
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      onAnalysisError('Please select an image first.');
      return;
    }

    onAnalysisStart();

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('subject', subject);

      const response = await axios.post<AnalysisData>(
        'http://localhost:8000/analyze-handwriting',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          timeout: 30000, // 30 second timeout
        }
      );

      if (response.data.success) {
        onAnalysisComplete(response.data);
      } else {
        onAnalysisError(response.data.error || 'Analysis failed.');
      }
    } catch (error) {
      console.error('Analysis error:', error);
      if (axios.isAxiosError(error)) {
        if (error.code === 'ECONNABORTED') {
          onAnalysisError('Request timeout. Please try again.');
        } else if (error.response) {
          onAnalysisError(error.response.data.detail || 'Server error occurred.');
        } else {
          onAnalysisError('Network error occurred.');
        }
      } else {
        onAnalysisError('An unknown error occurred.');
      }
    }
  };

  const handleCameraCapture = () => {
    // Camera capture feature (to be implemented)
    alert('Camera feature will be available in future updates.');
  };

  const clearSelection = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const subjectOptions = [
    { value: 'math', label: 'Math Problem', icon: Calculator },
    { value: 'essay', label: 'Essay/Summary', icon: FileText },
    { value: 'notes', label: 'Study Notes', icon: FileText },
  ];

  return (
    <div className="space-y-6">
      {/* Title and Description */}
      <div className="text-center">
        <h2 className="text-3xl font-bold text-gray-900 mb-4">
          Evaluate Math & Essays with AI
        </h2>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto">
          Upload handwritten math problems or essays. ThinkGrade will provide 
          step-by-step solutions and educational feedback with scoring.
        </p>
      </div>

      {/* Subject Selection */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Select content type to evaluate</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {subjectOptions.map((option) => {
            const Icon = option.icon;
            return (
              <button
                key={option.value}
                onClick={() => setSubject(option.value as SubjectType)}
                className={`p-4 rounded-lg border-2 transition-all duration-200 ${
                  subject === option.value
                    ? 'border-primary-500 bg-primary-50 text-primary-700'
                    : 'border-gray-200 hover:border-gray-300 text-gray-700'
                }`}
              >
                <Icon className="h-8 w-8 mx-auto mb-2" />
                <span className="font-medium">{option.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Image Upload Area */}
      <div className="card">
        {!previewUrl ? (
          <div
            className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-primary-400 transition-colors cursor-pointer"
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              Upload or drag an image here
            </h3>
            <p className="text-gray-600 mb-4">
              Supports JPG, PNG, GIF files (max 10MB)
            </p>
            <div className="flex justify-center space-x-4">
              <button
                type="button"
                className="btn-primary"
                onClick={(e) => {
                  e.stopPropagation();
                  fileInputRef.current?.click();
                }}
              >
                Choose File
              </button>
              <button
                type="button"
                className="btn-secondary"
                onClick={(e) => {
                  e.stopPropagation();
                  handleCameraCapture();
                }}
              >
                <Camera className="h-4 w-4 inline mr-2" />
                Camera
              </button>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="relative">
              <img
                src={previewUrl}
                alt="Uploaded image"
                className="w-full max-h-96 object-contain rounded-lg border border-gray-200"
              />
              <button
                onClick={clearSelection}
                className="absolute top-2 right-2 bg-red-500 text-white rounded-full p-2 hover:bg-red-600 transition-colors"
              >
                <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="flex justify-center space-x-4">
              <button
                onClick={handleAnalyze}
                disabled={isLoading}
                className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? 'Evaluating...' : 'Start Evaluation'}
              </button>
              <button
                onClick={clearSelection}
                className="btn-secondary"
                disabled={isLoading}
              >
                Choose Different Image
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileSelect}
        className="hidden"
      />
    </div>
  );
};

export default ImageUpload; 
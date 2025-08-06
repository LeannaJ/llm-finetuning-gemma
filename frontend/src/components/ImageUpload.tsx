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
      onAnalysisError('이미지 파일만 업로드 가능합니다.');
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
      onAnalysisError('이미지 파일만 업로드 가능합니다.');
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      onAnalysisError('이미지를 선택해주세요.');
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
          timeout: 30000, // 30초 타임아웃
        }
      );

      if (response.data.success) {
        onAnalysisComplete(response.data);
      } else {
        onAnalysisError(response.data.error || '분석에 실패했습니다.');
      }
    } catch (error) {
      console.error('분석 오류:', error);
      if (axios.isAxiosError(error)) {
        if (error.code === 'ECONNABORTED') {
          onAnalysisError('요청 시간이 초과되었습니다. 다시 시도해주세요.');
        } else if (error.response) {
          onAnalysisError(error.response.data.detail || '서버 오류가 발생했습니다.');
        } else {
          onAnalysisError('네트워크 오류가 발생했습니다.');
        }
      } else {
        onAnalysisError('알 수 없는 오류가 발생했습니다.');
      }
    }
  };

  const handleCameraCapture = () => {
    // 카메라 캡처 기능 (향후 구현)
    alert('카메라 기능은 향후 업데이트 예정입니다.');
  };

  const clearSelection = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const subjectOptions = [
    { value: 'math', label: 'Math', icon: Calculator },
    { value: 'essay', label: 'Essay', icon: FileText },
    { value: 'notes', label: 'Notes', icon: FileText },
  ];

  return (
    <div className="space-y-6">
      {/* Title and Description */}
      <div className="text-center">
        <h2 className="text-3xl font-bold text-gray-900 mb-4">
          Analyze Handwriting with AI
        </h2>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto">
          Take photos of math problems, essays, notes, and more. 
          AI will instantly analyze and provide educational feedback.
        </p>
      </div>

      {/* Subject Selection */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Select the subject to analyze</h3>
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
                {isLoading ? 'Analyzing...' : 'Start Analysis'}
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

      {/* 숨겨진 파일 입력 */}
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
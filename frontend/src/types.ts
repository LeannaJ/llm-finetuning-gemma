export interface AnalysisData {
  success: boolean;
  extracted_text: string;
  feedback: {
    analysis: string;
    score: number;
    suggestions: string[];
    summary: string;
  };
  subject: string;
}

export interface UploadResponse {
  success: boolean;
  extracted_text?: string;
  feedback?: {
    analysis: string;
    score: number;
    suggestions: string[];
    summary: string;
  };
  subject?: string;
  error?: string;
}

export type SubjectType = 'math' | 'essay' | 'notes'; 
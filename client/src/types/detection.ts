export interface DetectionResult {
  type: 'text' | 'image';
  result: 'human' | 'ai';
  confidence: number;
  timestamp: string;
  input?: string;
  filename?: string;
}

export interface DetectionHistory {
  id: string;
  userId: string;
  type: 'text' | 'image';
  result: 'human' | 'ai';
  confidence: number;
  input?: string;
  filename?: string;
  createdAt: string;
}
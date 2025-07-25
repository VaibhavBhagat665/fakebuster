export interface User {
  _id: string;
  name: string;
  email: string;
  password: string;
  createdAt: Date;
  updatedAt: Date;
}

export interface Detection {
  _id: string;
  userId: string;
  type: 'text' | 'image';
  result: 'human' | 'ai';
  confidence: number;
  input?: string;
  filename?: string;
  createdAt: Date;
}

export interface FlaskResponse {
  result: 'human' | 'ai';
  confidence: number;
}

export interface AuthRequest extends Request {
  user?: User;
}
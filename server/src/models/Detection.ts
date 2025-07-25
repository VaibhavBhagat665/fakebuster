import mongoose, { Schema, Document } from 'mongoose';

export interface IDetection extends Document {
  userId: mongoose.Types.ObjectId;
  type: 'text' | 'image';
  result: 'human' | 'ai';
  confidence: number;
  input?: string;
  filename?: string;
  createdAt: Date;
}

const detectionSchema = new Schema<IDetection>({
  userId: { type: Schema.Types.ObjectId, ref: 'User', required: true },
  type: { type: String, enum: ['text', 'image'], required: true },
  result: { type: String, enum: ['human', 'ai'], required: true },
  confidence: { type: Number, required: true, min: 0, max: 1 },
  input: { type: String },
  filename: { type: String },
}, { timestamps: true });

export const Detection = mongoose.model<IDetection>('Detection', detectionSchema);
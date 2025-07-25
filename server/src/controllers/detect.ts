import { Response } from 'express';
import { AuthRequest } from '../middleware/auth';
import { Detection } from '../models/Detection';
import { detectText, detectImage } from '../services/flask';

export const detectTextController = async (req: AuthRequest, res: Response) => {
  try {
    const { text } = req.body;
    const userId = req.user._id;

    const result = await detectText(text);
    
    const detection = new Detection({
      userId,
      type: 'text',
      result: result.result,
      confidence: result.confidence,
      input: text,
    });
    
    await detection.save();

    res.json({
      type: 'text',
      result: result.result,
      confidence: result.confidence,
      timestamp: detection.createdAt,
    });
  } catch (error) {
    console.error('Text detection error:', error);
    res.status(500).json({ message: 'Detection failed' });
  }
};

export const detectImageController = async (req: AuthRequest, res: Response) => {
  try {
    if (!req.file) {
      return res.status(400).json({ message: 'No image uploaded' });
    }

    const userId = req.user._id;
    const { buffer, originalname } = req.file;

    const result = await detectImage(buffer, originalname);
    
    const detection = new Detection({
      userId,
      type: 'image',
      result: result.result,
      confidence: result.confidence,
      filename: originalname,
    });
    
    await detection.save();

    res.json({
      type: 'image',
      result: result.result,
      confidence: result.confidence,
      filename: originalname,
      timestamp: detection.createdAt,
    });
  } catch (error) {
    console.error('Image detection error:', error);
    res.status(500).json({ message: 'Detection failed' });
  }
};
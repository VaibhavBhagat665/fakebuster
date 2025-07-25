import { Response } from 'express';
import { AuthRequest } from '../middleware/auth';
import { Detection } from '../models/Detection';

export const getHistory = async (req: AuthRequest, res: Response) => {
  try {
    const userId = req.user._id;
    const page = parseInt(req.query.page as string) || 1;
    const limit = parseInt(req.query.limit as string) || 20;
    const skip = (page - 1) * limit;

    const detections = await Detection.find({ userId })
      .sort({ createdAt: -1 })
      .skip(skip)
      .limit(limit)
      .select('-userId -__v');

    const total = await Detection.countDocuments({ userId });

    res.json({
      detections,
      pagination: {
        page,
        limit,
        total,
        pages: Math.ceil(total / limit),
      },
    });
  } catch (error) {
    console.error('History fetch error:', error);
    res.status(500).json({ message: 'Failed to fetch history' });
  }
};
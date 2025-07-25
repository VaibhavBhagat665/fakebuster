import { Router } from 'express';
import { getHistory } from '../controllers/history';
import { auth } from '../middleware/auth';

const router = Router();

router.get('/', auth, getHistory);

export default router;
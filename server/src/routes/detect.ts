import { Router } from 'express';
import { detectTextController, detectImageController } from '../controllers/detect';
import { auth } from '../middleware/auth';
import { upload } from '../middleware/upload';
import { textValidation, validateRequest } from '../middleware/validation';

const router = Router();

router.post('/text', auth, textValidation, validateRequest, detectTextController);
router.post('/image', auth, upload.single('image'), detectImageController);

export default router;
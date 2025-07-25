import { Router } from 'express';
import { register, login } from '../controllers/auth';
import { registerValidation, loginValidation, validateRequest } from '../middleware/validation';

const router = Router();

router.post('/register', registerValidation, validateRequest, register);
router.post('/login', loginValidation, validateRequest, login);

export default router;
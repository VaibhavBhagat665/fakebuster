import dotenv from 'dotenv';

dotenv.config();

export const config = {
  PORT: process.env.PORT || 5000,
  MONGODB_URI: process.env.MONGODB_URI || 'mongodb://localhost:27017/',
  JWT_SECRET: process.env.JWT_SECRET || 'your-super-secret-jwt-key',
  FLASK_URL: process.env.FLASK_URL || 'http://localhost:7000',
  NODE_ENV: process.env.NODE_ENV || 'development',
};
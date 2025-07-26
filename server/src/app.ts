import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import { connectDB } from './config/db';
import { config } from './config/env';

import authRoutes from './routes/auth';
import detectRoutes from './routes/detect';
import historyRoutes from './routes/history';

const app = express();

const getAllowedOrigins = (): (string | RegExp)[] => {
  const origins: (string | RegExp)[] = [
    'http://localhost:3000',
    'http://localhost:3001',
    'http://127.0.0.1:3000',
    'https://fakebuster-phi.vercel.app',
  ];

  if (process.env.CLIENT_URL) {
    origins.push(process.env.CLIENT_URL);
  }

  origins.push(/^https:\/\/fakebuster.*\.vercel\.app$/);
  origins.push(/^https:\/\/.*\.vercel\.app$/);

  return origins;
};

app.use(helmet({
  crossOriginResourcePolicy: { policy: "cross-origin" },
  crossOriginEmbedderPolicy: false
}));

app.use(cors({
  origin: function (origin: string | undefined, callback: (err: Error | null, allow?: boolean) => void) {
    const allowedOrigins = getAllowedOrigins();
    
    if (!origin) {
      return callback(null, true);
    }
    
    const isAllowed = allowedOrigins.some((allowedOrigin: string | RegExp) => {
      if (typeof allowedOrigin === 'string') {
        return allowedOrigin === origin;
      }
      return allowedOrigin.test(origin);
    });
    
    callback(null, isAllowed);
  },
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH'],
  allowedHeaders: [
    'Origin',
    'X-Requested-With',
    'Content-Type',
    'Accept',
    'Authorization',
    'Cache-Control',
    'X-HTTP-Method-Override'
  ],
  exposedHeaders: ['set-cookie'],
  optionsSuccessStatus: 200,
  preflightContinue: false
}));

app.use((req, res, next) => {
  const origin = req.headers.origin;
  if (origin) {
    res.header('Access-Control-Allow-Origin', origin);
  }
  res.header('Access-Control-Allow-Credentials', 'true');
  next();
});

app.use(express.json({ 
  limit: '10mb',
  strict: false
}));

app.use(express.urlencoded({ 
  extended: true, 
  limit: '10mb',
  parameterLimit: 1000
}));

app.options('*', (req, res) => {
  res.sendStatus(200);
});

app.use('/api/auth', authRoutes);
app.use('/api/detect', detectRoutes);
app.use('/api/history', historyRoutes);

app.get('/health', (req, res) => {
  res.status(200).json({ 
    status: 'OK', 
    timestamp: new Date().toISOString(),
    cors: 'enabled'
  });
});

app.use((err: any, req: express.Request, res: express.Response, next: express.NextFunction) => {
  console.error('Server Error:', err.message || err);
  
  if (err.message && err.message.includes('CORS')) {
    return res.status(403).json({ 
      message: 'CORS policy violation',
      origin: req.headers.origin 
    });
  }
  
  res.status(err.status || 500).json({ 
    message: err.message || 'Internal server error',
    ...(process.env.NODE_ENV === 'development' && { stack: err.stack })
  });
});

app.use('*', (req, res) => {
  res.status(404).json({ 
    message: 'Route not found',
    path: req.baseUrl 
  });
});

const startServer = async (): Promise<void> => {
  try {
    await connectDB();
    
    const port = parseInt(String(config.PORT || process.env.PORT || 3000), 10);
    
    app.listen(port, '0.0.0.0', () => {
      console.log(`Server: ${port}`);
      console.log(`Environment: ${config.NODE_ENV || 'development'}`);
      console.log(`CORS: Active`);
    });
  } catch (error) {
    console.error('Server Start Failed:', error);
    process.exit(1);
  }
};

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});

process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
  process.exit(1);
});

startServer();

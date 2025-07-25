import { DetectionResult } from '@/types/detection';

interface Props {
  result: DetectionResult;
}

export const ResultDisplay = ({ result }: Props) => {
  const isAI = result.result === 'ai';
  
  return (
    <div className={`p-4 rounded-md border ${isAI ? 'bg-red-50 border-red-200 dark:bg-red-900/20' : 'bg-green-50 border-green-200 dark:bg-green-900/20'}`}>
      <div className="flex items-center justify-between">
        <div>
          <h3 className={`font-semibold ${isAI ? 'text-red-700 dark:text-red-300' : 'text-green-700 dark:text-green-300'}`}>
            {isAI ? '🤖 AI Generated' : '👤 Human Created'}
          </h3>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Confidence: {(result.confidence * 100).toFixed(1)}%
          </p>
        </div>
        <div className={`text-2xl ${isAI ? 'text-red-500' : 'text-green-500'}`}>
          {isAI ? '⚠️' : '✅'}
        </div>
      </div>
    </div>
  );
};
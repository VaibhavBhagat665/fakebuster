'use client';
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { ResultDisplay } from './result-display';
import api from '@/lib/api';
import { DetectionResult } from '@/types/detection';

export const TextDetector = () => {
  const [text, setText] = useState('');
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [loading, setLoading] = useState(false);

  const handleDetect = async () => {
    if (!text.trim()) return;
    
    setLoading(true);
    try {
      const res = await api.post('/api/detect/text', { text });
      setResult(res.data);
    } catch (err) {
      console.error('Detection failed:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <div>
        <textarea
          className="w-full h-40 p-3 border border-gray-300 rounded-md resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-800"
          placeholder="Enter text to analyze..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
      </div>
      <Button onClick={handleDetect} disabled={loading || !text.trim()}>
        {loading ? 'Analyzing...' : 'Detect AI Text'}
      </Button>
      {result && <ResultDisplay result={result} />}
    </div>
  );
};
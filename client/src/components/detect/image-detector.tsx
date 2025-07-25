'use client';
import { useState, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { ResultDisplay } from './result-display';
import api from '@/lib/api';
import { DetectionResult } from '@/types/detection';

export const ImageDetector = () => {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      const reader = new FileReader();
      reader.onload = () => setPreview(reader.result as string);
      reader.readAsDataURL(selectedFile);
    }
  };

  const handleDetect = async () => {
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('image', file);

    try {
      const res = await api.post('/api/detect/image', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
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
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          ref={fileRef}
          className="hidden"
        />
        <Button 
          variant="outline" 
          onClick={() => fileRef.current?.click()}
          className="w-full"
        >
          Choose Image
        </Button>
      </div>
      
      {preview && (
        <div className="flex justify-center">
          <img 
            src={preview} 
            alt="Preview" 
            className="max-h-64 rounded-md object-contain"
          />
        </div>
      )}

      <Button onClick={handleDetect} disabled={loading || !file}>
        {loading ? 'Analyzing...' : 'Detect AI Image'}
      </Button>
      
      {result && <ResultDisplay result={result} />}
    </div>
  );
};
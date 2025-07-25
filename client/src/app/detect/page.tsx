'use client';
import { useState } from 'react';
import { useAuth } from '@/hooks/use-auth';
import { TextDetector } from '@/components/detect/text-detector';
import { ImageDetector } from '@/components/detect/image-detector';
import { Button } from '@/components/ui/button';
import { ThemeToggle } from '@/components/ui/theme-toggle';

export default function DetectPage() {
  const { user, logout } = useAuth();
  const [activeTab, setActiveTab] = useState<'text' | 'image'>('text');

  if (!user) {
    return <div>Loading...</div>;
  }

  return (
    <div className="min-h-screen p-4">
      <div className="max-w-4xl mx-auto">
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold">AI Detection</h1>
          <div className="flex items-center gap-4">
            <ThemeToggle />
            <Button variant="outline" onClick={logout}>
              Logout
            </Button>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md border border-gray-200 dark:border-gray-700">
          <div className="border-b border-gray-200 dark:border-gray-700">
            <nav className="flex space-x-8 px-6">
              <button
                onClick={() => setActiveTab('text')}
                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === 'text'
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
              >
                Text Detection
              </button>
              <button
                onClick={() => setActiveTab('image')}
                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === 'image'
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
              >
                Image Detection
              </button>
            </nav>
          </div>

          <div className="p-6">
            {activeTab === 'text' ? <TextDetector /> : <ImageDetector />}
          </div>
        </div>
      </div>
    </div>
  );
}
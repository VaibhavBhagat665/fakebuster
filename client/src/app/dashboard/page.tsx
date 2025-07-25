'use client';
import { useEffect, useState } from 'react';
import Link from 'next/link';
import { useAuth } from '@/hooks/use-auth';
import { Button } from '@/components/ui/button';
import { ThemeToggle } from '@/components/ui/theme-toggle';
import { DetectionHistory } from '@/types/detection';
import { formatDate } from '@/lib/utils';
import api from '@/lib/api';

export default function DashboardPage() {
  const { user, logout } = useAuth();
  const [history, setHistory] = useState<DetectionHistory[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (user) {
      fetchHistory();
    }
  }, [user]);

  const fetchHistory = async () => {
    try {
      const res = await api.get('/api/history?limit=10');
      setHistory(res.data.detections);
    } catch (error) {
      console.error('Failed to fetch history:', error);
    } finally {
      setLoading(false);
    }
  };

  if (!user) {
    return <div>Loading...</div>;
  }

  return (
    <div className="min-h-screen p-4">
      <div className="max-w-6xl mx-auto">
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-3xl font-bold">Dashboard</h1>
            <p className="text-gray-600 dark:text-gray-400">Welcome back, {user.name}!</p>
          </div>
          <div className="flex items-center gap-4">
            <ThemeToggle />
            <Button variant="outline" onClick={logout}>
              Logout
            </Button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          <Link href="/detect">
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md border border-gray-200 dark:border-gray-700 hover:shadow-lg transition-shadow cursor-pointer">
              <h3 className="text-xl font-semibold mb-2">🔍 AI Detection</h3>
              <p className="text-gray-600 dark:text-gray-400">
                Detect AI-generated text and images
              </p>
            </div>
          </Link>
          
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md border border-gray-200 dark:border-gray-700">
            <h3 className="text-xl font-semibold mb-2">📊 Statistics</h3>
            <p className="text-gray-600 dark:text-gray-400">
              Total detections: {history.length}
            </p>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md border border-gray-200 dark:border-gray-700">
          <div className="p-6 border-b border-gray-200 dark:border-gray-700">
            <h2 className="text-xl font-semibold">Recent Activity</h2>
          </div>
          
          <div className="p-6">
            {loading ? (
              <p>Loading history...</p>
            ) : history.length === 0 ? (
              <p className="text-gray-500">No detections yet. Start by analyzing some content!</p>
            ) : (
              <div className="space-y-4">
                {history.map((item) => (
                  <div key={item.id} className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                    <div className="flex items-center gap-4">
                      <div className="text-2xl">
                        {item.type === 'text' ? '📝' : '🖼️'}
                      </div>
                      <div>
                        <p className="font-medium">
                          {item.type === 'text' ? 'Text Analysis' : `Image: ${item.filename}`}
                        </p>
                        <p className="text-sm text-gray-500">
                          {formatDate(item.createdAt)}
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                        item.result === 'ai' 
                          ? 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-300' 
                          : 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-300'
                      }`}>
                        {item.result === 'ai' ? '🤖 AI' : '👤 Human'}
                      </div>
                      <p className="text-xs text-gray-500 mt-1">
                        {(item.confidence * 100).toFixed(1)}% confidence
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
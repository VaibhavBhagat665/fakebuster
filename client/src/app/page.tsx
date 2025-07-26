import Link from 'next/link';

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <div className="flex items-center justify-center min-h-screen p-4">
        <div className="text-center max-w-2xl">
          <h1 className="text-6xl font-bold mb-6 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            Auto Buster
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300 mb-8">
            Detect AI-generated content with advanced machine learning. 
            Analyze text and images to identify artificial intelligence patterns.
          </p>
          
          <div className="flex gap-4 justify-center">
            <Link href="/login">
              <button className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg transition-colors">
                Get Started
              </button>
            </Link>
            <Link href="/signup">
              <button className="border border-gray-300 hover:bg-gray-50 dark:border-gray-600 dark:hover:bg-gray-800 font-bold py-3 px-6 rounded-lg transition-colors">
                Sign Up
              </button>
            </Link>
          </div>
          
          <div className="mt-12 grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="p-6 bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
              <div className="text-3xl mb-4">📝</div>
              <h3 className="text-lg font-semibold mb-2">Text Detection</h3>
              <p className="text-gray-600 dark:text-gray-400">
                Analyze written content to identify AI-generated text with high accuracy
              </p>
            </div>
            
            <div className="p-6 bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
              <div className="text-3xl mb-4">🖼️</div>
              <h3 className="text-lg font-semibold mb-2">Image Detection</h3>
              <p className="text-gray-600 dark:text-gray-400">
                Identify AI-generated images and deepfakes using advanced computer vision
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

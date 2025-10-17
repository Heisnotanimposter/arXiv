import React, { useState } from 'react';

interface LoginProps {
  onLoginSuccess: () => void;
}

const Login: React.FC<LoginProps> = ({ onLoginSuccess }) => {
  const [id, setId] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleSocialLogin = (provider: 'Google' | 'Apple') => {
    console.log(`Simulating login with ${provider}...`);
    // In a real app, this would initiate the OAuth flow.
    // For this simulation, we'll just log the user in directly.
    onLoginSuccess();
  };

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError('');

    if (!id || !password) {
      setError('Both ID and password are required.');
      return;
    }

    // --- SIMULATED AUTHENTICATION ---
    // In a real application, this would be a secure API call to a backend.
    // The backend would handle password hashing and SQL injection prevention.
    // NEVER use hardcoded credentials like this in production.
    if (
      (id === 'admin' && password === 'password123') ||
      (id === '102416' && password === '614201')
    ) {
      onLoginSuccess();
    } else {
      setError('Invalid ID or password.');
    }
  };

  return (
    <div className="min-h-screen bg-primary text-text-base flex items-center justify-center animate-fade-in p-4">
      <div className="w-full max-w-sm">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-accent to-accent-2 font-serif">
            ArXiv Frontend Trends
          </h1>
          <p className="text-text-muted mt-2">
            Please sign in to continue.
          </p>
        </div>
        <div className="bg-secondary p-8 rounded-lg border border-border-color shadow-2xl shadow-primary/50">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-6">
            <button
              onClick={() => handleSocialLogin('Google')}
              className="flex items-center justify-center gap-2 w-full bg-tertiary hover:bg-tertiary/80 text-text-base font-semibold py-3 px-4 rounded-lg focus:outline-none focus:ring-4 focus:ring-tertiary/50 transition-all duration-300"
            >
              <svg className="w-5 h-5" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M44.5 24.3H42.7V24.3C42.7 23.3 42.6 22.3 42.4 21.4H24V27.2H35.3C34.7 29.2 33.3 30.9 31.4 32.2V36.2H37.3C41.7 32 44.5 26.8 44.5 24.3Z" fill="#4285F4"/>
                <path d="M24 45C30.2 45 35.5 42.8 39.2 39.2L33.2 34.7C31.1 36.1 27.9 37 24 37C17.1 37 11.2 32.5 9.1 26.5H3V31C6.7 39.1 14.7 45 24 45Z" fill="#34A853"/>
                <path d="M9.1 26.5C8.7 25.3 8.5 24.1 8.5 22.9C8.5 21.7 8.7 20.5 9.1 19.3V14.8H3C1.1 18.2 0 22.4 0 27C0 31.6 1.1 35.8 3 39.2L9.1 34.7V26.5Z" fill="#FBBC05"/>
                <path d="M24 8C27.9 8 31.1 9.4 33.6 11.6L39.5 5.7C35.5 2.1 30.2 0 24 0C14.7 0 6.7 5.9 3 14L9.1 18.5C11.2 12.5 17.1 8 24 8Z" fill="#EA4335"/>
              </svg>
              Google
            </button>
            <button
              onClick={() => handleSocialLogin('Apple')}
              className="flex items-center justify-center gap-2 w-full bg-tertiary hover:bg-tertiary/80 text-text-base font-semibold py-3 px-4 rounded-lg focus:outline-none focus:ring-4 focus:ring-tertiary/50 transition-all duration-300"
            >
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 16 16">
                <path d="M8.28,1.88a2.7,2.7,0,0,0-2.32,1.2,2.5,2.5,0,0,0,.8,3.52,2.41,2.41,0,0,0,3.12-.8A2.5,2.5,0,0,0,8.28,1.88Zm2.6,9.12a2.82,2.82,0,0,1-1.6,2.32,2.5,2.5,0,0,1-2.64-.6,8.25,8.25,0,0,1-1.36-2.24,6.44,6.44,0,0,1-2.4-4.64,3.22,3.22,0,0,1,2.8-3.28,2.74,2.74,0,0,1,2.56.8,1.25,1.25,0,0,0,1.68-.24,1.29,1.29,0,0,0,.24-1.68,5.46,5.46,0,0,0-4.4-2.4,5.65,5.65,0,0,0-5.6,5.84,8.5,8.5,0,0,0,2.64,6.4,7.5,7.5,0,0,0,5.2,2.48,5.52,5.52,0,0,0,4.8-2.64,1.25,1.25,0,0,0-2.16-1.2Z"/>
              </svg>
              Apple
            </button>
          </div>

          <div className="relative flex py-2 items-center">
            <div className="flex-grow border-t border-border-color"></div>
            <span className="flex-shrink mx-4 text-text-muted text-xs">OR</span>
            <div className="flex-grow border-t border-border-color"></div>
          </div>

          <form onSubmit={handleSubmit}>
            {error && (
              <div className="bg-danger-bg border border-danger-border text-danger-text px-4 py-3 rounded-lg my-4 text-center text-sm" role="alert">
                {error}
              </div>
            )}
            <div className="mt-4 mb-4">
              <label htmlFor="id" className="block text-text-muted text-sm font-bold mb-2">
                User ID
              </label>
              <input
                type="text"
                id="id"
                value={id}
                onChange={(e) => setId(e.target.value)}
                placeholder="e.g., 102416"
                className="w-full bg-tertiary/50 text-text-base placeholder-text-muted rounded-lg py-3 px-4 border border-border-color focus:outline-none focus:ring-2 focus:ring-accent-hover focus:border-transparent transition-all"
                autoComplete="username"
              />
            </div>
            <div className="mb-6">
              <label htmlFor="password" className="block text-text-muted text-sm font-bold mb-2">
                Password
              </label>
              <input
                type="password"
                id="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="e.g., 614201"
                className="w-full bg-tertiary/50 text-text-base placeholder-text-muted rounded-lg py-3 px-4 border border-border-color focus:outline-none focus:ring-2 focus:ring-accent-hover focus:border-transparent transition-all"
                autoComplete="current-password"
              />
            </div>
            <button
              type="submit"
              className="w-full bg-accent hover:bg-accent-hover text-text-inverted font-bold py-3 px-4 rounded-lg focus:outline-none focus:ring-4 focus:ring-accent/50 transition-all duration-300 transform hover:scale-105"
            >
              Sign In
            </button>
          </form>
        </div>
        <div className="text-center mt-6 text-xs text-text-muted/80">
          <p>This is a frontend simulation. No data is sent to a server.</p>
          <p>Defenses against threats like SQL injection are implemented on the backend.</p>
        </div>
      </div>
    </div>
  );
};

export default Login;
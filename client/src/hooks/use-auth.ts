'use client';
import { useState, useEffect } from 'react';
import { User } from '@/types/auth';
import { getUser, removeUser, removeAuthToken } from '@/lib/auth';

export const useAuth = () => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const storedUser = getUser();
    setUser(storedUser);
    setLoading(false);
  }, []);

  const logout = () => {
    removeAuthToken();
    removeUser();
    setUser(null);
    window.location.href = '/login';
  };

  return { user, loading, logout, setUser };
};
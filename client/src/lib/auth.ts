import Cookies from 'js-cookie';
import { User } from '@/types/auth';

export const setAuthToken = (token: string) => {
  Cookies.set('token', token, { expires: 7 });
};

export const removeAuthToken = () => {
  Cookies.remove('token');
};

export const getAuthToken = () => {
  return Cookies.get('token');
};

export const setUser = (user: User) => {
  localStorage.setItem('user', JSON.stringify(user));
};

export const getUser = (): User | null => {
  if (typeof window === 'undefined') return null;
  const user = localStorage.getItem('user');
  return user ? JSON.parse(user) : null;
};

export const removeUser = () => {
  localStorage.removeItem('user');
};
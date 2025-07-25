export interface User {
  id: string;
  email: string;
  name: string;
  createdAt: string;
}

export interface LoginReq {
  email: string;
  password: string;
}

export interface SignupReq {
  name: string;
  email: string;
  password: string;
}

export interface AuthRes {
  token: string;
  user: User;
}
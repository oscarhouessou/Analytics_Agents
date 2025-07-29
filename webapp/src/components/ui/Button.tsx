import React from 'react';
import clsx from 'clsx';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  icon?: React.ReactNode;
}

const base = 'inline-flex items-center justify-center font-medium rounded transition focus:outline-none focus:ring-2 focus:ring-offset-2';
const variants = {
  primary: 'bg-blue-600 text-white hover:bg-blue-700',
  secondary: 'bg-gray-100 text-gray-900 hover:bg-gray-200',
  ghost: 'bg-transparent text-blue-600 hover:bg-blue-50',
};
const sizes = {
  sm: 'px-3 py-1.5 text-sm',
  md: 'px-4 py-2 text-base',
  lg: 'px-6 py-3 text-lg',
};

export const Button: React.FC<ButtonProps> = ({
  children, variant = 'primary', size = 'md', icon, className, ...props
}) => (
  <button
    className={clsx(base, variants[variant], sizes[size], className)}
    {...props}
  >
    {icon && <span className="mr-2">{icon}</span>}
    {children}
  </button>
);

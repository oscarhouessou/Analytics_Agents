import * as React from 'react';
import { cn } from '../../utils/cn';

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost';
  loading?: boolean;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'primary', loading = false, disabled, children, ...props }, ref) => {
    return (
      <button
        ref={ref}
        className={cn(
          'inline-flex items-center justify-center rounded-md px-4 py-2 font-medium transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-blue-500 dark:focus-visible:ring-blue-400',
          variant === 'primary' && 'bg-blue-600 text-white hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600',
          variant === 'secondary' && 'bg-gray-100 text-gray-900 hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-100',
          variant === 'ghost' && 'bg-transparent hover:bg-gray-100 dark:hover:bg-gray-800',
          (loading || disabled) && 'opacity-60 pointer-events-none',
          className
        )}
        disabled={disabled || loading}
        aria-busy={loading}
        tabIndex={0}
        {...props}
      >
        {loading ? (
          <span className="animate-spin mr-2 h-4 w-4 border-2 border-t-transparent border-white rounded-full" aria-label="Chargement"></span>
        ) : null}
        {children}
      </button>
    );
  }
);
Button.displayName = 'Button';

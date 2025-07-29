import React from 'react';
import clsx from 'clsx';

export const Card: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({ className, ...props }) => (
  <div className={clsx('bg-white rounded-xl shadow p-6', className)} {...props} />
);

export const CardHeader: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({ className, ...props }) => (
  <div className={clsx('mb-4 flex items-center justify-between', className)} {...props} />
);

export const CardTitle: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({ className, ...props }) => (
  <div className={clsx('text-lg font-semibold', className)} {...props} />
);

export const CardContent: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({ className, ...props }) => (
  <div className={clsx('mt-2', className)} {...props} />
);

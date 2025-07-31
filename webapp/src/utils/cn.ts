// Utility to concatenate class names
export function cn(...args: (string | undefined | false | null)[]) {
  return args.filter(Boolean).join(' ')
}

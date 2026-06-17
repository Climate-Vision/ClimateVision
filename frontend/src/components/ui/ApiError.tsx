import { AlertCircle, X } from "lucide-react";


interface ApiErrorProps {
  message: string | null;
  onDismiss: () => void;
}

export function ApiError({ message, onDismiss }: ApiErrorProps) {
  if (!message) return null;

  return (
    <div role="alert" aria-live="assertive" className="w-full border border-red-500/40 bg-red-900/20 text-red-200 px-4 py-3 rounded-lg flex items-start justify-between gap-3">
      <div className="flex items-start gap-2">
        <AlertCircle className="w-5 h-5 mt-0.5" />

        <p className="text-sm break-words">{message}</p>
      </div>

      <button
        onClick={onDismiss}
        aria-label="Dismiss error"
        className="cursor-pointer hover:text-red-400 transition-colors"
      >
        <X className="w-4 h-4" />
      </button>
    </div>
  );
}
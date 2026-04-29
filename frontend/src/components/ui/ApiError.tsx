import { AlertCircle, X } from "lucide-react";
import { useState } from "react";

interface ApiErrorProps {
  message: string;
}

export function ApiError({ message }: ApiErrorProps) {
  const [visible, setVisible] = useState(true);

  if (!visible) return null;

  return (
    <div className="w-full border border-red-500/40 bg-red-900/20 text-red-200 px-4 py-3 rounded-lg flex items-start justify-between gap-3">
      <div className="flex items-start gap-2">
        <AlertCircle className="w-5 h-5 mt-0.5" />
        <p className="text-sm">{message}</p>
      </div>

      <button onClick={() => setVisible(false)}>
        <X className="w-4 h-4 hover:text-red-400" />
      </button>
    </div>
  );
}
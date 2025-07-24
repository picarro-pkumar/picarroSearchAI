import React, { useState, useRef, useEffect } from 'react';

interface MessageInputProps {
  onSendMessage: (message: string) => void;
  isLoading?: boolean;
  disabled?: boolean;
  placeholder?: string;
  isDarkMode?: boolean;
  onRegenerate?: () => void;
  canRegenerate?: boolean;
}

const MessageInput: React.FC<MessageInputProps> = ({ 
  onSendMessage, 
  isLoading = false,
  disabled = false, 
  placeholder = "Message Picarro SearchAI...",
  isDarkMode = true,
  onRegenerate,
  canRegenerate = false
}) => {
  const [message, setMessage] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() && !disabled && !isLoading) {
      onSendMessage(message.trim());
      setMessage('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const adjustTextareaHeight = () => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
    }
  };

  useEffect(() => {
    adjustTextareaHeight();
  }, [message]);

  return (
    <div className={`border-t ${isDarkMode ? 'border-chatgpt-gray-600 bg-chatgpt-gray-800' : 'border-gray-200 bg-white'}`}>
      <div className="max-w-3xl mx-auto p-4">
        <form onSubmit={handleSubmit} className="relative">
          <div className="relative">
            <textarea
              ref={textareaRef}
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={placeholder}
              disabled={disabled}
              className={`w-full resize-none rounded-lg border p-4 pr-12 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent min-h-[52px] max-h-[200px] transition-colors ${
                isDarkMode 
                  ? 'border-chatgpt-gray-600 bg-chatgpt-gray-700 text-chatgpt-gray-100 placeholder-chatgpt-gray-400' 
                  : 'border-gray-300 bg-gray-50 text-gray-900 placeholder-gray-500'
              }`}
              style={{ minHeight: '52px' }}
            />
            <button
              type="submit"
              disabled={!message.trim() || disabled || isLoading}
              className={`absolute right-2 bottom-2 p-2 rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                isDarkMode 
                  ? 'text-chatgpt-gray-400 hover:text-chatgpt-gray-200 hover:bg-chatgpt-gray-600' 
                  : 'text-gray-500 hover:text-gray-700 hover:bg-gray-200'
              }`}
            >
              <svg 
                className="w-5 h-5" 
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24"
              >
                <path 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  strokeWidth={2} 
                  d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" 
                />
              </svg>
            </button>
          </div>
          
          {canRegenerate && onRegenerate && (
            <div className="mt-2 flex justify-center">
              <button
                onClick={onRegenerate}
                disabled={isLoading}
                className={`px-4 py-2 text-sm rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                  isDarkMode 
                    ? 'text-chatgpt-gray-400 hover:text-chatgpt-gray-200 hover:bg-chatgpt-gray-700' 
                    : 'text-gray-600 hover:text-gray-800 hover:bg-gray-100'
                }`}
              >
                🔄 Regenerate response
              </button>
            </div>
          )}
          
          <div className={`mt-2 text-xs text-center ${isDarkMode ? 'text-chatgpt-gray-400' : 'text-gray-500'}`}>
            Picarro SearchAI can make mistakes. Consider checking important information.
          </div>
        </form>
      </div>
    </div>
  );
};

export default MessageInput; 
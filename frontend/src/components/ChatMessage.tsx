import React, { useState } from 'react';
import { Message } from '../types';

interface ChatMessageProps {
  message: Message;
  onRegenerate?: () => void;
  isDarkMode?: boolean;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message, onRegenerate, isDarkMode = true }) => {
  const [copied, setCopied] = useState(false);
  const [showSources, setShowSources] = useState(false);

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  };

  const formatTimestamp = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const renderSources = () => {
    if (!message.sources || message.sources.length === 0) return null;

    return (
      <div className={`mt-4 border-t pt-4 ${isDarkMode ? 'border-chatgpt-gray-600' : 'border-gray-200'}`}>
        <button
          onClick={() => setShowSources(!showSources)}
          className={`text-sm mb-2 transition-colors ${isDarkMode ? 'text-chatgpt-gray-400 hover:text-chatgpt-gray-200' : 'text-gray-500 hover:text-gray-700'}`}
        >
          {showSources ? 'Hide' : 'Show'} References ({message.sources.length})
        </button>
        
        {showSources && (
          <div className="space-y-3">
            {message.sources.map((source, index) => (
              <div key={index} className={`rounded-lg p-3 overflow-hidden ${isDarkMode ? 'bg-chatgpt-gray-700' : 'bg-gray-100'}`}>
                <div className="flex justify-between items-start mb-2">
                  <span className={`text-xs ${isDarkMode ? 'text-chatgpt-gray-400' : 'text-gray-500'}`}>
                    Source {index + 1} (Score: {source.similarity_score.toFixed(3)})
                  </span>
                  {source.metadata.product && (
                    <span className={`text-xs px-2 py-1 rounded ${isDarkMode ? 'bg-chatgpt-gray-600' : 'bg-gray-200'}`}>
                      {source.metadata.product}
                    </span>
                  )}
                </div>
                
                {/* Show Confluence link if available */}
                {source.metadata.page_url ? (
                  <div className="mb-3">
                    <a 
                      href={source.metadata.page_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className={`inline-flex items-center space-x-2 text-sm font-medium ${isDarkMode ? 'text-blue-400 hover:text-blue-300' : 'text-blue-600 hover:text-blue-700'} transition-colors`}
                    >
                      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                        <path d="M11 3a1 1 0 100 2h2.586l-6.293 6.293a1 1 0 101.414 1.414L15 6.414V9a1 1 0 102 0V4a1 1 0 00-1-1h-5z" />
                        <path d="M5 5a2 2 0 00-2 2v8a2 2 0 002 2h8a2 2 0 002-2v-3a1 1 0 10-2 0v3H5V7h3a1 1 0 000-2H5z" />
                      </svg>
                      <span>View in Confluence</span>
                    </a>
                    {source.metadata.title && (
                      <p className={`text-xs mt-1 ${isDarkMode ? 'text-chatgpt-gray-400' : 'text-gray-600'}`}>
                        {source.metadata.title}
                      </p>
                    )}
                    {/* Additional metadata */}
                    <div className={`mt-2 text-xs space-y-1 ${isDarkMode ? 'text-chatgpt-gray-400' : 'text-gray-500'}`}>
                      {source.metadata.space_name && (
                        <div>Space: {source.metadata.space_name}</div>
                      )}
                      {source.metadata.author && (
                        <div>Author: {source.metadata.author}</div>
                      )}
                      {source.metadata.last_modified && (
                        <div>Updated: {new Date(source.metadata.last_modified).toLocaleDateString()}</div>
                      )}
                    </div>
                  </div>
                ) : (
                  /* Fallback to content preview if no URL */
                  <p className={`text-sm leading-relaxed source-content ${isDarkMode ? 'text-chatgpt-gray-200' : 'text-gray-800'}`}>
                    {source.content.length > 200 ? `${source.content.substring(0, 200)}...` : source.content}
                  </p>
                )}
                
                {source.metadata.category && (
                  <div className={`mt-2 text-xs ${isDarkMode ? 'text-chatgpt-gray-400' : 'text-gray-500'}`}>
                    Category: {source.metadata.category}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    );
  };

  if (message.role === 'user') {
    return (
      <div className="flex justify-center py-6">
        <div className="flex items-start space-x-4 max-w-3xl w-full px-4">
          <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${isDarkMode ? 'bg-chatgpt-gray-600' : 'bg-gray-500'}`}>
            <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clipRule="evenodd" />
            </svg>
          </div>
          <div className="flex-1 min-w-0">
            <div className={`rounded-lg p-4 ${isDarkMode ? 'bg-chatgpt-gray-700' : 'bg-gray-100'}`}>
              <p className={`leading-relaxed whitespace-pre-wrap ${isDarkMode ? 'text-chatgpt-gray-100' : 'text-gray-800'}`}>
                {message.content}
              </p>
            </div>
            <div className="flex items-center justify-between mt-2">
              <span className={`text-xs ${isDarkMode ? 'text-chatgpt-gray-400' : 'text-gray-500'}`}>
                {formatTimestamp(message.timestamp)}
              </span>
              <button
                onClick={copyToClipboard}
                className={`text-xs opacity-0 group-hover:opacity-100 transition-opacity ${isDarkMode ? 'text-chatgpt-gray-400 hover:text-chatgpt-gray-200' : 'text-gray-500 hover:text-gray-700'}`}
              >
                {copied ? 'Copied!' : 'Copy'}
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`flex justify-center py-6 ${isDarkMode ? 'bg-chatgpt-gray-800' : 'bg-gray-50'}`}>
      <div className="flex items-start space-x-4 max-w-3xl w-full px-4">
        <div className="w-8 h-8 rounded-full bg-green-600 flex items-center justify-center flex-shrink-0">
          <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 20 20">
            <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </div>
        <div className="flex-1 min-w-0">
          <div className="bg-transparent">
            {message.isTyping ? (
              <div className="flex items-center space-x-2">
                <div className="typing-dots">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            ) : (
              <>
                <div className="prose prose-invert max-w-none">
                  <p className={`leading-relaxed whitespace-pre-wrap ${isDarkMode ? 'text-chatgpt-gray-100' : 'text-gray-800'}`}>
                    {message.content}
                  </p>
                </div>
                {renderSources()}
              </>
            )}
          </div>
          <div className="flex items-center justify-between mt-2">
            <div className="flex items-center space-x-4">
              <span className={`text-xs ${isDarkMode ? 'text-chatgpt-gray-400' : 'text-gray-500'}`}>
                {formatTimestamp(message.timestamp)}
              </span>
              {/* LLM Response Time and Model Info */}
              {message.response_time && (
                <div className={`text-xs ${isDarkMode ? 'text-chatgpt-gray-400' : 'text-gray-500'}`}>
                  <span className="font-medium">LLM:</span> {message.response_time.toFixed(1)}s 
                  {message.model_used && (
                    <span className="ml-1">({message.model_used})</span>
                  )}
                  <span className="ml-2 text-xs opacity-75">
                    {message.response_time > 10 ? 'Processing large context' : 
                     message.response_time > 5 ? 'Analyzing sources' : 'Quick response'}
                  </span>
                </div>
              )}
              {/* Confidence Score */}
              {message.confidence_score && (
                <div className={`text-xs ${isDarkMode ? 'text-chatgpt-gray-400' : 'text-gray-500'}`}>
                  <span className="font-medium">Confidence:</span> {(message.confidence_score * 100).toFixed(0)}%
                </div>
              )}
            </div>
            <div className="flex items-center space-x-2">
              <button
                onClick={copyToClipboard}
                className={`text-xs opacity-0 group-hover:opacity-100 transition-opacity ${isDarkMode ? 'text-chatgpt-gray-400 hover:text-chatgpt-gray-200' : 'text-gray-500 hover:text-gray-700'}`}
              >
                {copied ? 'Copied!' : 'Copy'}
              </button>
              {onRegenerate && (
                <button
                  onClick={onRegenerate}
                  className={`text-xs opacity-0 group-hover:opacity-100 transition-opacity ${isDarkMode ? 'text-chatgpt-gray-400 hover:text-chatgpt-gray-200' : 'text-gray-500 hover:text-gray-700'}`}
                >
                  Regenerate
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatMessage; 
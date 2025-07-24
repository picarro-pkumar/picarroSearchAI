import React, { useState } from 'react';
import { Chat } from '../types';

interface SidebarProps {
  chats: Chat[];
  currentChatId: string | null;
  onNewChat: () => void;
  onSelectChat: (chatId: string) => void;
  onDeleteChat: (chatId: string) => void;
  onClearAll: () => void;
  onUpdateTitle: (chatId: string, title: string) => void;
  isOpen: boolean;
  onToggle: () => void;
  isDarkMode: boolean;
  onToggleTheme: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({
  chats,
  currentChatId,
  onNewChat,
  onSelectChat,
  onDeleteChat,
  onClearAll,
  onUpdateTitle,
  isOpen,
  onToggle,
  isDarkMode,
  onToggleTheme
}) => {
  const [hoveredChatId, setHoveredChatId] = useState<string | null>(null);
  const [showClearConfirm, setShowClearConfirm] = useState(false);

  const handleClearAllChats = () => {
    if (showClearConfirm) {
      onClearAll();
      setShowClearConfirm(false);
    } else {
      setShowClearConfirm(true);
      // Auto-hide confirmation after 3 seconds
      setTimeout(() => setShowClearConfirm(false), 3000);
    }
  };

  const formatDate = (date: Date) => {
    const now = new Date();
    const diffInHours = (now.getTime() - date.getTime()) / (1000 * 60 * 60);
    
    if (diffInHours < 24) {
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } else if (diffInHours < 168) { // 7 days
      return date.toLocaleDateString([], { weekday: 'short' });
    } else {
      return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
    }
  };

  const truncateTitle = (title: string) => {
    return title.length > 30 ? title.substring(0, 30) + '...' : title;
  };

  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
          onClick={onToggle}
        />
      )}

      {/* Sidebar */}
      <div className={`
        fixed top-0 left-0 h-full w-80 transform transition-transform duration-150 ease-out z-50
        ${isDarkMode ? 'bg-chatgpt-gray-900 border-r border-chatgpt-gray-600' : 'bg-white border-r border-gray-200'}
        ${isOpen ? 'translate-x-0' : '-translate-x-full'}
        lg:translate-x-0 lg:static lg:z-auto
      `}>
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className={`flex items-center justify-between p-4 border-b ${isDarkMode ? 'border-chatgpt-gray-600' : 'border-gray-200'}`}>
            <button
              onClick={onNewChat}
              className={`flex items-center space-x-2 w-full p-3 rounded-lg border transition-colors ${isDarkMode ? 'border-chatgpt-gray-600 text-chatgpt-gray-200 hover:bg-chatgpt-gray-700' : 'border-gray-300 text-gray-700 hover:bg-gray-100'}`}
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              <span>New chat</span>
            </button>
            
            {/* Mobile close button */}
            <button
              onClick={onToggle}
              className={`lg:hidden ml-2 p-2 rounded-lg transition-colors ${isDarkMode ? 'text-chatgpt-gray-400 hover:text-chatgpt-gray-200 hover:bg-chatgpt-gray-700' : 'text-gray-600 hover:text-gray-800 hover:bg-gray-100'}`}
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Chat list */}
          <div className="flex-1 overflow-y-auto">
            {chats.length === 0 ? (
              <div className={`p-4 text-center ${isDarkMode ? 'text-chatgpt-gray-400' : 'text-gray-500'}`}>
                <p>No conversations yet</p>
                <p className="text-sm mt-1">Start a new chat to begin</p>
              </div>
            ) : (
              <div className="p-2">
                {chats.map((chat) => (
                  <div
                    key={chat.id}
                    className={`
                      group relative flex items-center space-x-3 p-3 rounded-lg cursor-pointer transition-colors
                      ${currentChatId === chat.id 
                        ? (isDarkMode ? 'bg-chatgpt-gray-700 text-chatgpt-gray-200' : 'bg-gray-100 text-gray-900')
                        : (isDarkMode ? 'text-chatgpt-gray-300 hover:bg-chatgpt-gray-800 hover:text-chatgpt-gray-200' : 'text-gray-700 hover:bg-gray-50 hover:text-gray-900')
                      }
                    `}
                    onClick={() => onSelectChat(chat.id)}
                    onMouseEnter={() => setHoveredChatId(chat.id)}
                    onMouseLeave={() => setHoveredChatId(null)}
                  >
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate">
                        {truncateTitle(chat.title)}
                      </p>
                      <p className={`text-xs truncate ${isDarkMode ? 'text-chatgpt-gray-400' : 'text-gray-500'}`}>
                        {formatDate(chat.updatedAt)}
                      </p>
                    </div>
                    
                    {/* Delete button */}
                    {hoveredChatId === chat.id && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onDeleteChat(chat.id);
                        }}
                        className={`opacity-0 group-hover:opacity-100 p-1 rounded transition-all ${isDarkMode ? 'text-chatgpt-gray-400 hover:text-red-400 hover:bg-chatgpt-gray-700' : 'text-gray-500 hover:text-red-500 hover:bg-gray-100'}`}
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                      </button>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Footer */}
          <div className={`p-4 border-t space-y-2 ${isDarkMode ? 'border-chatgpt-gray-600' : 'border-gray-200'}`}>
            {/* Clear all chats button */}
            {chats.length > 0 && (
              <button
                onClick={handleClearAllChats}
                className={`w-full p-2 text-sm rounded-lg transition-colors ${
                  showClearConfirm 
                    ? 'bg-red-600 text-white hover:bg-red-700' 
                    : (isDarkMode ? 'text-red-400 hover:text-red-300 hover:bg-chatgpt-gray-700' : 'text-red-500 hover:text-red-600 hover:bg-gray-100')
                }`}
              >
                {showClearConfirm ? 'Click again to confirm' : 'Clear all chats'}
              </button>
            )}
            
            <div className={`flex items-center space-x-3 p-3 rounded-lg ${isDarkMode ? 'bg-chatgpt-gray-800' : 'bg-gray-100'}`}>
              <div className="w-8 h-8 rounded-full bg-green-600 flex items-center justify-center">
                <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <div className="flex-1 min-w-0">
                <p className={`text-sm font-medium ${isDarkMode ? 'text-chatgpt-gray-200' : 'text-gray-800'}`}>Picarro SearchAI</p>
                <p className={`text-xs ${isDarkMode ? 'text-chatgpt-gray-400' : 'text-gray-600'}`}>Cloud Fenceline Project</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default Sidebar; 
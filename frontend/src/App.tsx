import React, { useState, useEffect, useRef } from 'react';
import ChatMessage from './components/ChatMessage';
import MessageInput from './components/MessageInput';
import Sidebar from './components/Sidebar';
import { Message, Chat, SearchResponse } from './types';
import './App.css';

const API_BASE_URL = 'http://localhost:8000';

// API service functions
const apiService = {
  async saveChat(chatId: string | null, messages: Message[], title?: string): Promise<{ chatId: string; title: string }> {
    const response = await fetch(`${API_BASE_URL}/api/save-chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        chat_id: chatId,
        title,
        messages: messages.map(msg => ({
          id: msg.id,
          content: msg.content,
          role: msg.role,
          timestamp: msg.timestamp.toISOString(),
          sources: msg.sources,
          isTyping: msg.isTyping
        }))
      })
    });
    
    if (!response.ok) {
      throw new Error(`Failed to save chat: ${response.statusText}`);
    }
    
    const data = await response.json();
    return { chatId: data.chat_id, title: data.title };
  },

  async loadChatHistory(): Promise<Chat[]> {
    const response = await fetch(`${API_BASE_URL}/api/chat-history`);
    
    if (!response.ok) {
      throw new Error(`Failed to load chat history: ${response.statusText}`);
    }
    
    const data = await response.json();
    return data.chats.map((chat: any) => ({
      id: chat.id,
      title: chat.title,
      messages: chat.messages.map((msg: any) => ({
        ...msg,
        timestamp: new Date(msg.timestamp)
      })),
      createdAt: new Date(chat.created_at),
      updatedAt: new Date(chat.updated_at)
    }));
  },

  async loadChat(chatId: string): Promise<Chat> {
    const response = await fetch(`${API_BASE_URL}/api/chat/${chatId}`);
    
    if (!response.ok) {
      throw new Error(`Failed to load chat: ${response.statusText}`);
    }
    
    const chat = await response.json();
    return {
      id: chat.id,
      title: chat.title,
      messages: chat.messages.map((msg: any) => ({
        ...msg,
        timestamp: new Date(msg.timestamp)
      })),
      createdAt: new Date(chat.created_at),
      updatedAt: new Date(chat.updated_at)
    };
  },

  async deleteChat(chatId: string): Promise<void> {
    const response = await fetch(`${API_BASE_URL}/api/chat/${chatId}`, {
      method: 'DELETE'
    });
    
    if (!response.ok) {
      throw new Error(`Failed to delete chat: ${response.statusText}`);
    }
  },

  async updateChatTitle(chatId: string, title: string): Promise<void> {
    const response = await fetch(`${API_BASE_URL}/api/chat/${chatId}/title`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title })
    });
    
    if (!response.ok) {
      throw new Error(`Failed to update chat title: ${response.statusText}`);
    }
  }
};

function App() {
  const [chats, setChats] = useState<Chat[]>([]);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(() => {
    const savedTheme = localStorage.getItem('picarro-theme');
    return savedTheme ? savedTheme === 'dark' : true; // Default to dark mode
  });
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingMessage, setProcessingMessage] = useState('');
  const [processingStage, setProcessingStage] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Save theme preference to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem('picarro-theme', isDarkMode ? 'dark' : 'light');
  }, [isDarkMode]);

  // Check for ongoing processing on page load
  useEffect(() => {
    const savedProcessingState = localStorage.getItem('picarro-processing-state');
    if (savedProcessingState) {
      try {
        const state = JSON.parse(savedProcessingState);
        const timeDiff = Date.now() - state.timestamp;
        // Only restore if processing was started within the last 5 minutes
        if (timeDiff < 5 * 60 * 1000) {
          setIsProcessing(true);
          setProcessingMessage(state.message);
          setProcessingStage(state.stage);
          console.log('🔄 Restored processing state from page refresh');
        } else {
          localStorage.removeItem('picarro-processing-state');
        }
      } catch (error) {
        console.error('Error parsing saved processing state:', error);
        localStorage.removeItem('picarro-processing-state');
      }
    }
  }, []);

  // Load chat history from backend on component mount
  useEffect(() => {
    const loadChatHistory = async () => {
      try {
        console.log('🔄 Loading chat history from backend...');
        const chatHistory = await apiService.loadChatHistory();
        setChats(chatHistory);
        console.log('✅ Chat history loaded:', chatHistory.length, 'chats');
      } catch (error) {
        console.error('❌ Error loading chat history:', error);
        setError('Failed to load chat history');
      } finally {
        setIsLoadingHistory(false);
      }
    };

    loadChatHistory();
  }, []);

  // Auto-save chat to backend whenever messages change
  useEffect(() => {
    if (messages.length > 0 && !isLoadingHistory) {
      const saveChat = async () => {
        try {
          const title = messages[0]?.content?.slice(0, 50) || 'New Chat';
          const { chatId, title: savedTitle } = await apiService.saveChat(currentChatId, messages, title);
          
          // Update current chat ID if it was null (new chat)
          if (!currentChatId) {
            setCurrentChatId(chatId);
          }
          
          // Update chat in the list
          setChats(prevChats => {
            const existingChatIndex = prevChats.findIndex(chat => chat.id === chatId);
            const updatedChat = {
              id: chatId,
              title: savedTitle,
              messages,
              createdAt: existingChatIndex >= 0 ? prevChats[existingChatIndex].createdAt : new Date(),
              updatedAt: new Date()
            };
            
            if (existingChatIndex >= 0) {
              const newChats = [...prevChats];
              newChats[existingChatIndex] = updatedChat;
              return newChats;
            } else {
              return [updatedChat, ...prevChats];
            }
          });
          
          console.log('💾 Chat auto-saved to backend');
        } catch (error) {
          console.error('❌ Error auto-saving chat:', error);
          setError('Failed to save chat');
        }
      };

      // Debounce auto-save to avoid too many API calls
      const timeoutId = setTimeout(saveChat, 1000);
      return () => clearTimeout(timeoutId);
    }
  }, [messages, currentChatId, isLoadingHistory]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const createNewChat = () => {
    console.log('🆕 Creating new chat');
    setCurrentChatId(null);
    setMessages([]);
    setError(null);
    setSidebarOpen(false);
  };

  const selectChat = async (chatId: string) => {
    try {
      console.log('📂 Loading chat:', chatId);
      const chat = await apiService.loadChat(chatId);
      setCurrentChatId(chatId);
      setMessages(chat.messages);
      setError(null);
      setSidebarOpen(false);
      console.log('✅ Chat loaded:', chat.title);
    } catch (error) {
      console.error('❌ Error loading chat:', error);
      setError('Failed to load chat');
    }
  };

  const deleteChat = async (chatId: string) => {
    try {
      await apiService.deleteChat(chatId);
      setChats(prevChats => prevChats.filter(chat => chat.id !== chatId));
      
      // If we're deleting the current chat, start a new one
      if (currentChatId === chatId) {
        createNewChat();
      }
      
      console.log('🗑️ Chat deleted:', chatId);
    } catch (error) {
      console.error('❌ Error deleting chat:', error);
      setError('Failed to delete chat');
    }
  };

  const clearAllChats = async () => {
    try {
      // Delete all chats from backend
      await Promise.all(chats.map(chat => apiService.deleteChat(chat.id)));
      setChats([]);
      createNewChat();
      console.log('🧹 All chats cleared');
    } catch (error) {
      console.error('❌ Error clearing chats:', error);
      setError('Failed to clear chats');
    }
  };

  const updateChatTitle = async (chatId: string, newTitle: string) => {
    try {
      await apiService.updateChatTitle(chatId, newTitle);
      setChats(prevChats => 
        prevChats.map(chat => 
          chat.id === chatId ? { ...chat, title: newTitle } : chat
        )
      );
      console.log('✏️ Chat title updated:', newTitle);
    } catch (error) {
      console.error('❌ Error updating chat title:', error);
      setError('Failed to update chat title');
    }
  };

  const processingStages = [
    { message: 'Analyzing your question...', icon: '🔍', color: '#3B82F6' },
    { message: 'Searching knowledge base...', icon: '📚', color: '#10B981' },
    { message: 'Processing information...', icon: '⚡', color: '#F59E0B' },
    { message: 'Generating insights...', icon: '💡', color: '#8B5CF6' },
    { message: 'Crafting response...', icon: '✍️', color: '#EF4444' },
    { message: 'Finalizing answer...', icon: '✨', color: '#06B6D4' }
  ];

  const startProcessingAnimation = () => {
    let stage = 0;
    const interval = setInterval(() => {
      setProcessingStage(stage);
      setProcessingMessage(processingStages[stage].message);
      
      // Update localStorage with current processing state
      localStorage.setItem('picarro-processing-state', JSON.stringify({
        message: processingStages[stage].message,
        stage: stage,
        timestamp: Date.now()
      }));
      
      stage = (stage + 1) % processingStages.length;
    }, 1200); // Cycle through stages every 1.2 seconds
    
    return interval;
  };

  const sendMessage = async (content: string) => {
    if (!content.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: content.trim(),
      role: 'user',
      timestamp: new Date(),
      isTyping: false
    };

          setMessages(prev => [...prev, userMessage]);
      setIsLoading(true);
      setIsProcessing(true);
      setProcessingStage(0);
      setProcessingMessage(processingStages[0].message);
      setError(null);

      // Save processing state to localStorage for page refresh recovery
      localStorage.setItem('picarro-processing-state', JSON.stringify({
        message: processingStages[0].message,
        stage: 0,
        timestamp: Date.now()
      }));

      const processingInterval = startProcessingAnimation();

    try {
      const response = await fetch(`${API_BASE_URL}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: content.trim() })
      });

      if (!response.ok) {
        throw new Error(`Search failed: ${response.statusText}`);
      }

      const data: SearchResponse = await response.json();

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: data.answer,
        role: 'assistant',
        timestamp: new Date(),
        sources: data.sources,
        isTyping: false
      };

      setMessages(prev => [...prev, assistantMessage]);
      console.log('✅ Message sent and response received');
    } catch (error) {
      console.error('❌ Error sending message:', error);
      setError('Failed to send message. Please try again.');
    } finally {
      if (processingInterval) {
        clearInterval(processingInterval);
      }
      setIsLoading(false);
      setIsProcessing(false);
      setProcessingMessage('');
      setProcessingStage(0);
      // Clear processing state from localStorage
      localStorage.removeItem('picarro-processing-state');
    }
  };

  const regenerateResponse = async () => {
    if (messages.length === 0) return;

    const lastUserMessage = messages.slice().reverse().find(msg => msg.role === 'user');
    if (!lastUserMessage) return;

    // Remove the last assistant message if it exists
    const messagesWithoutLastAssistant = messages.filter((msg, index) => {
      if (msg.role === 'assistant' && index === messages.length - 1) {
        return false;
      }
      return true;
    });

    setMessages(messagesWithoutLastAssistant);
    setIsLoading(true);
    setIsProcessing(true);
    setProcessingStage(0);
    setProcessingMessage(processingStages[0].message);
    setError(null);

    // Save processing state to localStorage for page refresh recovery
    localStorage.setItem('picarro-processing-state', JSON.stringify({
      message: processingStages[0].message,
      stage: 0,
      timestamp: Date.now()
    }));

    const processingInterval = startProcessingAnimation();

    try {
      const response = await fetch(`${API_BASE_URL}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: lastUserMessage.content })
      });

      if (!response.ok) {
        throw new Error(`Search failed: ${response.statusText}`);
      }

      const data: SearchResponse = await response.json();

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: data.answer,
        role: 'assistant',
        timestamp: new Date(),
        sources: data.sources,
        isTyping: false
      };

      setMessages(prev => [...prev, assistantMessage]);
      console.log('🔄 Response regenerated');
    } catch (error) {
      console.error('❌ Error regenerating response:', error);
      setError('Failed to regenerate response. Please try again.');
    } finally {
      if (processingInterval) {
        clearInterval(processingInterval);
      }
      setIsLoading(false);
      setIsProcessing(false);
      setProcessingMessage('');
      setProcessingStage(0);
      // Clear processing state from localStorage
      localStorage.removeItem('picarro-processing-state');
    }
  };

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode);
  };

  if (isLoadingHistory) {
    return (
      <div className={`app ${isDarkMode ? 'dark' : 'light'}`}>
        <div className="loading-screen">
          <div className="loading-spinner"></div>
          <p>Loading chat history...</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`app ${isDarkMode ? 'dark' : 'light'}`}>
      <Sidebar
        chats={chats}
        currentChatId={currentChatId}
        isOpen={sidebarOpen}
        onToggle={toggleSidebar}
        onSelectChat={selectChat}
        onDeleteChat={deleteChat}
        onNewChat={createNewChat}
        onClearAll={clearAllChats}
        onUpdateTitle={updateChatTitle}
        isDarkMode={isDarkMode}
        onToggleTheme={toggleTheme}
      />

      <div className="main-content">
        <header className="header">
          <button className="sidebar-toggle" onClick={toggleSidebar}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="3" y1="6" x2="21" y2="6"></line>
              <line x1="3" y1="12" x2="21" y2="12"></line>
              <line x1="3" y1="18" x2="21" y2="18"></line>
            </svg>
          </button>
          <h1>Picarro SearchAI</h1>
          <button className="theme-toggle" onClick={toggleTheme}>
            {isDarkMode ? '☀️' : '🌙'}
          </button>
        </header>

        <main className="chat-container">
          {error && (
            <div className="error-message">
              <span>{error}</span>
              <button onClick={() => setError(null)}>✕</button>
            </div>
          )}

          <div className="messages">
            {messages.length === 0 ? (
              <div className="welcome-message">
                <h2>Welcome to Picarro SearchAI</h2>
                <div className="fenceline-note">
                  <p><strong>Note:</strong> Picarro SearchAI is currently designed for Fenceline Cloud Solution</p>
                </div>
                <div className="example-queries">
                  <p>Try asking:</p>
                  <ul>
                    <li>"What are Picarro's main products?"</li>
                    <li>"How does cavity ring-down spectroscopy work?"</li>
                    <li>"Tell me about Picarro's environmental monitoring solutions"</li>
                  </ul>
                </div>
              </div>
            ) : (
              messages.map((message) => (
                <ChatMessage
                  key={message.id}
                  message={message}
                  isDarkMode={isDarkMode}
                />
              ))
            )}
            
            {isProcessing && (
              <div className="processing-message">
                <div className="processing-stage">
                  <div className="processing-icon-container">
                    <span 
                      className="processing-icon"
                      style={{ color: processingStages[processingStage].color }}
                    >
                      {processingStages[processingStage].icon}
                    </span>
                    <div className="processing-ripple"></div>
                  </div>
                  <div className="processing-progress">
                    <div className="progress-bar">
                      <div 
                        className="progress-fill"
                        style={{ 
                          width: `${((processingStage + 1) / processingStages.length) * 100}%`,
                          backgroundColor: processingStages[processingStage].color
                        }}
                      ></div>
                    </div>
                  </div>
                </div>
                <div className="processing-content">
                  <span 
                    className="processing-text"
                    style={{ color: processingStages[processingStage].color }}
                  >
                    {processingMessage}
                  </span>
                  <div className="processing-particles">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>

          <div className="input-container">
            <MessageInput
              onSendMessage={sendMessage}
              isLoading={isLoading}
              isDarkMode={isDarkMode}
              onRegenerate={regenerateResponse}
              canRegenerate={messages.length > 0 && messages[messages.length - 1]?.role === 'assistant'}
            />
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;

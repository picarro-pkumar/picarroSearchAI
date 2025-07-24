export interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
  sources?: Source[];
  isTyping?: boolean;
}

export interface Source {
  content: string;
  metadata: Record<string, any>;
  similarity_score: number;
  rank: number;
}

export interface SearchResponse {
  answer: string;
  sources: Source[];
  query: string;
  response_time: number;
  confidence_score?: number;
  model_used: string;
  timestamp: string;
}

export interface SearchQuery {
  query: string;
  max_results?: number;
  filter_metadata?: Record<string, any>;
}

export interface Chat {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
} 
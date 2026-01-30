"""
QA Chain module for question answering with memory and context retrieval.
Handles retrieval-augmented generation with conversation memory.
"""

import logging
from typing import List, Dict, Tuple
from datetime import datetime

import openai
from embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Manages short-term conversation history (session-only).
    """
    
    def __init__(self, max_messages: int = 10):
        """
        Initialize conversation memory.
        
        Args:
            max_messages: Maximum number of messages to keep in memory
        """
        self.messages: List[Dict] = []
        self.max_messages = max_messages
    
    def add_message(self, role: str, content: str):
        """
        Add a message to conversation history.
        
        Args:
            role: "user" or "assistant"
            content: Message content
        """
        self.messages.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only recent messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_context(self) -> List[Dict]:
        """
        Get conversation history for context.
        
        Returns:
            List of message dictionaries
        """
        return self.messages
    
    def clear(self):
        """Clear conversation history."""
        self.messages = []
    
    def get_summary(self) -> str:
        """
        Get a summary of recent conversation.
        
        Returns:
            String summary of recent messages
        """
        if not self.messages:
            return ""
        
        summary_parts = []
        for msg in self.messages[-4:]:  # Last 2 exchanges
            prefix = "User: " if msg['role'] == 'user' else "Assistant: "
            summary_parts.append(prefix + msg['content'][:100] + "...")
        
        return "\n".join(summary_parts)


class QAChain:
    """
    Question-Answering chain with retrieval and memory.
    
    Features:
    - Semantic search in indexed content
    - LLM-powered answer generation
    - Conversation memory
    - Strict grounding in website content
    """
    
    SYSTEM_PROMPT = """You are a helpful AI assistant. Answer questions STRICTLY based on the provided website content.

IMPORTANT RULES:
1. Only use information from the retrieved content below
2. If the answer is not in the content, respond EXACTLY with: "The answer is not available on the provided website."
3. Do not make up or assume any information
4. Do not use external knowledge
5. Cite the source when relevant

Retrieved Content:
{context}

Answer the user's question based ONLY on this content."""
    
    def __init__(self, embedding_manager: EmbeddingManager, api_key: str):
        """
        Initialize QA Chain.
        
        Args:
            embedding_manager: Initialized EmbeddingManager instance
            api_key: OpenAI API key
        """
        self.embedding_manager = embedding_manager
        self.memory = ConversationMemory(max_messages=20)
        
        # Configure OpenAI
        openai.api_key = api_key
        self.model = "gpt-3.5-turbo"
        self.temperature = 0.7
        self.max_tokens = 500
    
    def _retrieve_context(self, query: str, k: int = 5) -> Tuple[str, List[Dict]]:
        """
        Retrieve relevant content chunks for a query.
        
        Args:
            query: User question
            k: Number of chunks to retrieve
            
        Returns:
            Tuple of (formatted context string, list of source chunks)
        """
        results = self.embedding_manager.search(query, k=k)
        
        if not results:
            return "", []
        
        context_parts = []
        for idx, result in enumerate(results, 1):
            context_parts.append(f"[Source {idx}]:\n{result['content']}")
        
        context = "\n\n".join(context_parts)
        return context, results
    
    def _format_messages(self, query: str, context: str) -> List[Dict]:
        """
        Format messages for OpenAI API.
        
        Args:
            query: User question
            context: Retrieved context
            
        Returns:
            List of message dictionaries
        """
        system_prompt = self.SYSTEM_PROMPT.format(context=context)
        
        messages = [
            {'role': 'system', 'content': system_prompt}
        ]
        
        # Add conversation history (skip system messages)
        for msg in self.memory.messages:
            if msg['role'] != 'system':
                messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
        
        # Add current query
        messages.append({'role': 'user', 'content': query})
        
        return messages
    
    def answer(self, query: str) -> Tuple[str, List[Dict]]:
        """
        Answer a user question based on indexed website content.
        
        Args:
            query: User question
            
        Returns:
            Tuple of (answer text, source chunks used)
        """
        # Add user message to memory
        self.memory.add_message('user', query)
        
        # Retrieve relevant content
        context, sources = self._retrieve_context(query, k=5)
        
        if not context:
            answer = "The answer is not available on the provided website."
            self.memory.add_message('assistant', answer)
            return answer, []
        
        # Prepare messages for LLM
        messages = self._format_messages(query, context)
        
        try:
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            answer = response.choices[0].message['content'].strip()
            
            # Add assistant response to memory
            self.memory.add_message('assistant', answer)
            
            logger.info(f"Generated answer for query: {query[:50]}...")
            return answer, sources
            
        except openai.error.APIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            answer = "Error generating answer. Please try again."
            return answer, []
        except openai.error.AuthenticationError:
            logger.error("Invalid OpenAI API key")
            answer = "Authentication error. Please check your API key."
            return answer, []
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            answer = "An error occurred while generating the answer."
            return answer, []
    
    def reset_memory(self):
        """Reset conversation memory."""
        self.memory.clear()
        logger.info("Conversation memory reset")
    
    def get_memory_summary(self) -> str:
        """Get summary of conversation memory."""
        return self.memory.get_summary()


def create_qa_chain(embedding_manager: EmbeddingManager, api_key: str) -> QAChain:
    """Factory function to create a QAChain instance."""
    return QAChain(embedding_manager, api_key)
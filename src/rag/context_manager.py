# Makes sure we dont exceed token limits when adding retrieved docs

class ContextManager:
    def __init__(self, max_tokens=2000, chars_per_token=4):
        """
        max_tokens: maximum tokens to use for context
        chars_per_token: rough estimate (English is ~4 chars per token)
        """
        self.max_tokens = max_tokens
        self.chars_per_token = chars_per_token
        self.max_chars = max_tokens * chars_per_token
    
    def build_context(self, retrieved_results, include_scores=False):
        """
        Build context string from retrieved chunks, fitting within token limit
        retrieved_results: list of dicts with 'chunk' and 'score'
        """
        if not retrieved_results:
            return ""
        
        context_parts = []
        current_size = 0
        
        for result in retrieved_results:
            chunk = result['chunk']
            chunk_text = chunk['text']
            chunk_size = len(chunk_text)
            
            # check if adding this chunk would exceed limit
            if current_size + chunk_size > self.max_chars:
                # try to fit partial chunk
                remaining = self.max_chars - current_size
                if remaining > 200:  # only add if we can fit meaningful content
                    chunk_text = chunk_text[:remaining] + "..."
                    chunk_size = len(chunk_text)
                else:
                    break
            
            # format chunk with header
            header = chunk.get('header', 'Documentation')
            chunk_context = f"\n--- {header} ---\n{chunk_text}\n"
            
            if include_scores:
                score = result.get('score', 0)
                chunk_context = f"\n--- {header} (relevance: {score:.3f}) ---\n{chunk_text}\n"
            
            context_parts.append(chunk_context)
            current_size += len(chunk_context)
            
            if current_size >= self.max_chars:
                break
        
        return "".join(context_parts)
    
    def build_rag_prompt(self, user_query, context, base_prompt="", max_total_tokens=4000):
        """
        Combine user query with retrieved context into final prompt
        Ensures total prompt fits within LLM token limit
        """
        if not context:
            # no context available, return basic prompt
            return base_prompt + f"\n\nUser request: {user_query}\n"
        
        # estimate tokens for base prompt and query
        base_tokens = self.estimate_tokens(base_prompt + f"\n\nUser request: {user_query}\n")
        context_tokens = self.estimate_tokens(context)
        total_tokens = base_tokens + context_tokens
        
        # if too long, truncate context
        if total_tokens > max_total_tokens:
            available_tokens = max_total_tokens - base_tokens - 100  # leave room for formatting
            if available_tokens > 200:  # minimum useful context
                context = self.truncate_to_limit(context, available_tokens)
            else:
                context = ""  # skip context if not enough room
        
        prompt = base_prompt + "\n\n"
        prompt += " RELEVANT DOCUMENTATION \n"
        prompt += context
        prompt += "\n END DOCUMENTATION \n\n"
        prompt += f"User request: {user_query}\n\n"
        prompt += "Use the documentation above to generate accurate Manim code. "
        prompt += "Follow the documented APIs and patterns.\n"
        
        return prompt
    
    def estimate_tokens(self, text):
        # Rough token count estimate
        return len(text) // self.chars_per_token
    
    def truncate_to_limit(self, text):
        # Truncate text to fit within token limit
        if len(text) <= self.max_chars:
            return text
        
        return text[:self.max_chars] + "\n\n[Content truncated to fit token limit]"

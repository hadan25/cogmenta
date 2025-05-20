# cogmenta_core/interface/conversation.py

import time
import json
import os
from datetime import datetime

class ConversationInterface:
    """
    Interface for natural language interaction with the cognitive architecture.
    """
    
    def __init__(self, cognitive_controller):
        """
        Initialize the conversation interface.
        
        Args:
            cognitive_controller: The central cognitive controller
        """
        self.controller = cognitive_controller
        self.conversation_history = []
        self.session_id = f"session_{int(time.time())}"
        
        # Create sessions directory if it doesn't exist
        if not os.path.exists("sessions"):
            os.makedirs("sessions")
    
    def process_input(self, user_message):
        """
        Process user input and generate a response.
        
        Args:
            user_message: User's input message
            
        Returns:
            System's response
        """
        # Create message record
        user_message_record = {
            "role": "user",
            "content": user_message,
            "timestamp": time.time()
        }
        self.conversation_history.append(user_message_record)
        
        # Process message through cognitive controller
        context = self._build_context()
        processing_start = time.time()
        result = self.controller.process_input(user_message, context=context)
        processing_time = time.time() - processing_start
        
        # Extract the response from result
        if isinstance(result, dict) and "response" in result:
            response_content = result["response"]
        elif isinstance(result, dict) and "certain" in result:
            # Convert knowledge facts to natural language
            response_content = self._facts_to_natural_language(result)
        else:
            # Default response if result format is unexpected
            response_content = f"I processed your input: '{user_message}'"
            if isinstance(result, dict):
                response_content += f"\nResult: {json.dumps(result, indent=2)}"
            else:
                response_content += f"\nResult: {result}"
        
        # Create response record
        system_response_record = {
            "role": "system",
            "content": response_content,
            "timestamp": time.time(),
            "processing_time": processing_time,
            "internal_state": self._capture_internal_state()
        }
        self.conversation_history.append(system_response_record)
        
        # Save the conversation after each exchange
        self._save_conversation()
        
        return response_content
    
    def _build_context(self):
        """
        Build context from conversation history for current processing.
        
        Returns:
            Context dictionary
        """
        # Extract recent messages
        recent_messages = self.conversation_history[-10:] if len(self.conversation_history) > 10 else self.conversation_history
        
        # Build context object
        context = {
            "conversation_history": recent_messages,
            "message_count": len(self.conversation_history),
            "session_duration": time.time() - self.conversation_history[0]["timestamp"] if self.conversation_history else 0
        }
        
        return context
    
    def _facts_to_natural_language(self, facts_result):
        """
        Convert structured knowledge facts to natural language.
        
        Args:
            facts_result: Result with certain/uncertain facts
            
        Returns:
            Natural language response
        """
        response_parts = []
        
        # Process certain facts
        if "certain" in facts_result and facts_result["certain"]:
            response_parts.append("Here's what I know:")
            for fact in facts_result["certain"]:
                subj = fact.get("subject", "unknown").replace("_", " ").title()
                pred = fact.get("predicate", "is related to").replace("_", " ")
                obj = fact.get("object", "unknown").replace("_", " ").title()
                confidence = fact.get("confidence", 0)
                
                if confidence > 0.8:
                    confidence_str = "definitely"
                elif confidence > 0.6:
                    confidence_str = "likely"
                else:
                    confidence_str = "possibly"
                
                sentence = f"{subj} {pred} {obj} ({confidence_str})"
                response_parts.append(sentence)
        
        # Process uncertain facts
        if "uncertain" in facts_result and facts_result["uncertain"]:
            response_parts.append("\nI'm less certain about these:")
            for fact in facts_result["uncertain"]:
                subj = fact.get("subject", "unknown").replace("_", " ").title()
                pred = fact.get("predicate", "is related to").replace("_", " ")
                obj = fact.get("object", "unknown").replace("_", " ").title()
                
                sentence = f"{subj} might {pred} {obj}"
                response_parts.append(sentence)
        
        # If no facts, create a default response
        if not response_parts:
            response_parts.append("I don't have enough information to provide a response.")
        
        return "\n".join(response_parts)
    
    def _capture_internal_state(self):
        """
        Capture relevant internal state for logging.
        
        Returns:
            Dictionary with internal state data
        """
        internal_state = {}
        
        # Capture state from cognitive controller if available
        if hasattr(self.controller, 'get_status'):
            internal_state["system_status"] = self.controller.get_status()
            
        return internal_state
    
    def _save_conversation(self):
        """Save conversation history to a file"""
        # Create a filename based on session ID
        filename = f"sessions/{self.session_id}.json"
        
        # Convert timestamps to readable format for storage
        formatted_history = []
        for message in self.conversation_history:
            message_copy = message.copy()
            timestamp = message_copy.pop("timestamp")
            message_copy["time"] = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            formatted_history.append(message_copy)
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(formatted_history, f, indent=2)
    
    def get_conversation_summary(self):
        """
        Get a summary of the current conversation.
        
        Returns:
            Summary dictionary
        """
        if not self.conversation_history:
            return {"status": "No conversation yet", "message_count": 0}
            
        user_messages = [m for m in self.conversation_history if m["role"] == "user"]
        system_messages = [m for m in self.conversation_history if m["role"] == "system"]
        
        # Calculate average response time
        if system_messages:
            avg_processing_time = sum(m.get("processing_time", 0) for m in system_messages) / len(system_messages)
        else:
            avg_processing_time = 0
            
        return {
            "message_count": len(self.conversation_history),
            "user_message_count": len(user_messages),
            "system_message_count": len(system_messages),
            "avg_processing_time": avg_processing_time,
            "session_id": self.session_id,
            "duration_minutes": (time.time() - self.conversation_history[0]["timestamp"])/60 if self.conversation_history else 0
        }
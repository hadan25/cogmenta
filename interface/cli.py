# cogmenta_core/interface/cli.py

import cmd
import time
import sys
import os
from datetime import datetime

class CogmentaShell(cmd.Cmd):
    """
    Command-line interface for interacting with Cogmenta Core.
    """
    
    intro = """
    =================================================================
    Welcome to Cogmenta Core - Neuro-Symbolic Cognitive Architecture
    =================================================================
    Type 'help' for a list of commands or just start chatting!
    """
    prompt = "You: "
    
    def __init__(self, conversation_interface):
        """
        Initialize the CLI.
        
        Args:
            conversation_interface: ConversationInterface instance
        """
        super().__init__()
        self.conversation = conversation_interface
        self.start_time = time.time()
        
    def default(self, line):
        """Handle any input not matching a command as conversation"""
        if line.strip():
            print("\nCogmenta is thinking...")
            response = self.conversation.process_input(line)
            print(f"\nCogmenta: {response}\n")
    
    def do_exit(self, arg):
        """Exit the Cogmenta CLI"""
        duration = time.time() - self.start_time
        minutes = int(duration / 60)
        seconds = int(duration % 60)
        
        summary = self.conversation.get_conversation_summary()
        print(f"\nSession summary:")
        print(f"- Duration: {minutes} minutes, {seconds} seconds")
        print(f"- Messages exchanged: {summary.get('message_count', 0)}")
        print(f"- Average processing time: {summary.get('avg_processing_time', 0):.2f} seconds")
        print(f"- Session saved as: {summary.get('session_id', 'unknown')}.json")
        print("\nThank you for using Cogmenta Core!\n")
        return True
        
    def do_status(self, arg):
        """Display system status"""
        if hasattr(self.conversation.controller, 'get_status'):
            status = self.conversation.controller.get_status()
            print("\nSystem Status:")
            print(f"- Components: {', '.join([c for c, v in status.get('components', {}).items() if v])}")
            print(f"- Processing history: {status.get('processing_history', 0)} items")
            print(f"- Phase 2 readiness: {status.get('ready_for_phase2', {}).get('overall_ready', False)}")
        else:
            print("\nStatus information not available")
    
    def do_history(self, arg):
        """Display conversation history"""
        history = self.conversation.conversation_history
        
        if not history:
            print("\nNo conversation history yet.")
            return
            
        print("\nConversation History:")
        for i, message in enumerate(history):
            role = message["role"]
            time_str = datetime.fromtimestamp(message["timestamp"]).strftime('%H:%M:%S')
            
            if i % 2 == 0:  # Add extra line between exchanges
                print()
                
            print(f"[{time_str}] {role.capitalize()}: {message['content']}")
    
    do_quit = do_exit
    do_bye = do_exit
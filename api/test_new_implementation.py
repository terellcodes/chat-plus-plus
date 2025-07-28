#!/usr/bin/env python3
"""
Test script for the new session-based implementation.
"""

import asyncio
import tempfile
import os
from services.document import DocumentService
from services.retrieval import RetrievalService


async def test_implementation():
    """Test the new implementation with a simple text file."""
    
    # Create a simple test PDF content
    test_content = """
This is a test document for the new implementation.
It contains multiple paragraphs to test the chunking and retrieval functionality.

The document discusses various topics including:
- Machine learning algorithms
- Natural language processing
- Information retrieval systems

This implementation uses session-based caching for better performance.
"""
    
    # Create a temporary "PDF" file (we'll just write text for testing)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdf', delete=False) as tmp_file:
        tmp_file.write(test_content)
        temp_path = tmp_file.name
    
    try:
        print("🧪 Testing new session-based implementation...")
        
        # Test document service
        print("\n1. Testing document upload and caching...")
        document_service = DocumentService()
        
        # Simulate file upload
        with open(temp_path, 'rb') as f:
            result = await document_service.process_pdf(
                f, 
                "test_document.pdf"
            )
        
        print(f"✅ Upload result: {result}")
        session_id = result.session_id
        
        # Test retrieval service
        print(f"\n2. Testing lazy retriever creation for session {session_id}...")
        retrieval_service = RetrievalService()
        
        # Test getting session info
        session_info = await retrieval_service.get_session_info(session_id)
        print(f"📊 Session info: {session_info}")
        
        # Test listing available strategies
        strategies = await retrieval_service.list_available_strategies()
        print(f"📋 Available strategies: {strategies}")
        
        print("\n✅ Basic implementation test completed successfully!")
        print(f"Session ID: {session_id}")
        
        return session_id
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        raise
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


if __name__ == "__main__":
    asyncio.run(test_implementation())
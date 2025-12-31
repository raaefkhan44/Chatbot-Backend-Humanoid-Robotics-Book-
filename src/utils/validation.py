from typing import Tuple
import re


def validate_question_content(content: str) -> Tuple[bool, str]:
    """
    Validate question content according to requirements
    - Must be between 5 and 1000 characters
    """
    if len(content) < 5:
        return False, "Question content must be at least 5 characters long"

    if len(content) > 1000:
        return False, "Question content must be no more than 1000 characters long"

    return True, "Valid question content"


def validate_book_content_chunk(content: str) -> Tuple[bool, str]:
    """
    Validate book content chunk according to requirements
    - Must be between 50 and 2000 characters
    """
    if len(content) < 50:
        return False, "Book content chunk must be at least 50 characters long"

    if len(content) > 2000:
        return False, "Book content chunk must be no more than 2000 characters long"

    return True, "Valid book content chunk"


def validate_session_id(session_id: str) -> Tuple[bool, str]:
    """
    Validate session ID according to requirements
    - Must follow the format /[a-zA-Z0-9-_]+/
    """
    if not re.match(r'^[a-zA-Z0-9-_]+$', session_id):
        return False, "Session ID must contain only alphanumeric characters, hyphens, and underscores"

    return True, "Valid session ID"


def validate_source_mode(source_mode: str) -> Tuple[bool, str]:
    """
    Validate source mode according to requirements
    - Must be either 'full' or 'selected'
    """
    if source_mode not in ['full', 'selected']:
        return False, "Source mode must be either 'full' or 'selected'"

    return True, "Valid source mode"


def validate_selected_text_length(selected_text: str) -> Tuple[bool, str]:
    """
    Validate selected text length
    - Should be at least 10 characters for meaningful context
    - Should be no more than 5000 characters to prevent API limits
    """
    if len(selected_text) < 10:
        return False, "Selected text must be at least 10 characters long for meaningful context"

    if len(selected_text) > 5000:
        return False, "Selected text must be no more than 5000 characters long to prevent API limits"

    return True, "Valid selected text length"
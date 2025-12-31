from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()


class LogEntry(Base):
    """
    Entity: Log Entry
    Description: Record of user interaction including question, response, and metadata
    """
    __tablename__ = "log_entries"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    question_id = Column(String, ForeignKey("questions.id"), nullable=False)  # Reference to the question
    answer_id = Column(String, ForeignKey("answers.id"), nullable=False)  # Reference to the answer
    user_session = Column(String, nullable=True)  # User session identifier
    mode = Column(String(20), nullable=False)  # Operation mode ('full' or 'selected')
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)  # When the interaction occurred
    user_feedback = Column(Text, nullable=True)  # Optional user feedback on the answer quality

    # Relationships
    question = relationship("Question")  # Reference to the question (many-to-one)
    answer = relationship("Answer", back_populates="log_entry")  # Reference to the answer (one-to-one)

    def __repr__(self):
        return f"<LogEntry(id={self.id}, mode={self.mode}, timestamp={self.timestamp})>"
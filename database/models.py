from datetime import datetime
from sqlalchemy import BigInteger, String, DateTime, ForeignKey, Integer, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.ext.asyncio import AsyncAttrs

class Base(AsyncAttrs, DeclarativeBase):
    pass

# Таблица пользователей [cite: 79, 80]
class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    telegram_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    username: Mapped[str] = mapped_column(String, nullable=True)
    full_name: Mapped[str] = mapped_column(String, nullable=True)
    registration_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_activity: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Связь с диалогами
    conversations = relationship("Conversation", back_populates="user")

# Таблица диалогов [cite: 81, 82]
class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    message_text: Mapped[str] = mapped_column(Text)
    bot_response: Mapped[str] = mapped_column(Text)
    message_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    tokens_used: Mapped[int] = mapped_column(Integer, default=0) # [cite: 82]

    # Связь с пользователем
    user = relationship("User", back_populates="conversations")

# Добавь это в database/models.py
class AdminDocument(Base):
    __tablename__ = "admin_documents"
    id: Mapped[int] = mapped_column(primary_key=True)
    document_name: Mapped[str] = mapped_column(String)
    file_path: Mapped[str] = mapped_column(String)
    file_hash: Mapped[str] = mapped_column(String, unique=True) # Для избежания дублей
    document_type: Mapped[str] = mapped_column(String)
    upload_date: Mapped[datetime] = mapped_column(DateTime)

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("admin_documents.id"))
    chunk_index: Mapped[int] = mapped_column(Integer)
    chunk_text: Mapped[str] = mapped_column(Text)
    embedding_id: Mapped[str] = mapped_column(String) # UUID в Qdrant
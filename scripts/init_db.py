#!/usr/bin/env python3

import os
import sys
import pyodbc
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get the project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Load environment variables
try:
    from dotenv import load_dotenv
    env_path = ROOT_DIR / 'config' / '.env'
    load_dotenv(env_path)
    logger.info(f"Loaded environment from {env_path}")
except Exception as e:
    logger.error(f"Error loading environment: {e}")
    sys.exit(1)

def create_database_connection():
    """Create a connection to the SQL Server database."""
    try:
        conn_str = (
            'DRIVER={ODBC Driver 18 for SQL Server};'
            f'SERVER={os.getenv("DB_HOST", "localhost")};'
            f'DATABASE={os.getenv("DB_NAME", "AI")};'
            f'UID={os.getenv("DB_USER", "aiuser")};'
            f'PWD={os.getenv("DB_PASSWORD")};'
            'TrustServerCertificate=yes;'
            'Encrypt=no;'
        )
        
        connection = pyodbc.connect(conn_str)
        logger.info("Successfully connected to the database")
        return connection
    except pyodbc.Error as e:
        logger.error(f"Error connecting to database: {e}")
        raise

def create_tables(connection):
    """Create necessary database tables."""
    try:
        cursor = connection.cursor()
        
        # Create face embeddings table
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='tblFaceEmbeddings' AND xtype='U')
            CREATE TABLE tblFaceEmbeddings (
                FaceID VARCHAR(36) PRIMARY KEY,
                PersonName NVARCHAR(100) NOT NULL,
                FaceEmbeddings VARBINARY(MAX) NOT NULL,
                Created_at DATETIME DEFAULT GETDATE()
            )
        """)
        
        # Create authentication logs table
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='tblAuthLogs' AND xtype='U')
            CREATE TABLE tblAuthLogs (
                LogID VARCHAR(36) PRIMARY KEY,
                SessionID VARCHAR(100) NOT NULL,
                AuthResult BIT NOT NULL,
                Confidence FLOAT,
                Message NVARCHAR(200),
                Created_at DATETIME DEFAULT GETDATE()
            )
        """)
        
        # Create sessions table
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='tblSessions' AND xtype='U')
            CREATE TABLE tblSessions (
                SessionID VARCHAR(100) PRIMARY KEY,
                ClientIP VARCHAR(45) NOT NULL,
                UserAgent NVARCHAR(500),
                Status VARCHAR(20) NOT NULL,
                Created_at DATETIME DEFAULT GETDATE(),
                LastUpdate DATETIME,
                ExpiresAt DATETIME
            )
        """)
        
        connection.commit()
        logger.info("Successfully created database tables")
        
    except pyodbc.Error as e:
        logger.error(f"Error creating tables: {e}")
        raise

def create_indexes(connection):
    """Create necessary indexes for performance."""
    try:
        cursor = connection.cursor()
        
        # Indexes for face embeddings table
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_FaceEmbeddings_PersonName')
            CREATE INDEX IX_FaceEmbeddings_PersonName ON tblFaceEmbeddings(PersonName)
        """)
        
        # Indexes for auth logs table
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_AuthLogs_SessionID')
            CREATE INDEX IX_AuthLogs_SessionID ON tblAuthLogs(SessionID)
        """)
        
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_AuthLogs_Created_at')
            CREATE INDEX IX_AuthLogs_Created_at ON tblAuthLogs(Created_at)
        """)
        
        # Indexes for sessions table
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_Sessions_Status')
            CREATE INDEX IX_Sessions_Status ON tblSessions(Status)
        """)
        
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_Sessions_ClientIP')
            CREATE INDEX IX_Sessions_ClientIP ON tblSessions(ClientIP)
        """)
        
        connection.commit()
        logger.info("Successfully created database indexes")
        
    except pyodbc.Error as e:
        logger.error(f"Error creating indexes: {e}")
        raise

def main():
    """Initialize the database."""
    try:
        # Create database connection
        connection = create_database_connection()
        
        # Create tables
        create_tables(connection)
        
        # Create indexes
        create_indexes(connection)
        
        logger.info("Database initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        sys.exit(1)
    finally:
        if 'connection' in locals():
            connection.close()

if __name__ == "__main__":
    main() 
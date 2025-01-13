import pyodbc
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

def connect_to_database():
    """Create a connection to the SQL Server database."""
    conn_str = 'DRIVER={ODBC Driver 18 for SQL Server};SERVER=172.16.15.161;DATABASE=AI;UID=aiuser;PWD=AIP@ss0rdSQL;TrustServerCertificate=yes;Encrypt=no;'
    
    try:
        connection = pyodbc.connect(conn_str)
        logging.info("Successfully connected to the database!")
        return connection
    except pyodbc.Error as e:
        logging.error(f"Error connecting to the database: {e}")
        raise

def delete_duplicates():
    """Delete duplicate faces from the database."""
    try:
        # Connect to database
        connection = connect_to_database()
        cursor = connection.cursor()
        
        # First, let's see what we have
        cursor.execute("SELECT PersonName, COUNT(*) as count FROM tblFaceEmbeddings GROUP BY PersonName")
        counts = cursor.fetchall()
        
        print("\nCurrent face counts:")
        for name, count in counts:
            print(f"{name}: {count} faces")
            
        # Delete duplicates keeping the most recent entry for each person
        delete_query = """
        WITH DuplicateRanking AS (
            SELECT 
                FaceID,
                PersonName,
                ROW_NUMBER() OVER (PARTITION BY PersonName ORDER BY Created_at DESC) as rn
            FROM tblFaceEmbeddings
        )
        DELETE FROM tblFaceEmbeddings
        WHERE FaceID IN (
            SELECT FaceID 
            FROM DuplicateRanking 
            WHERE rn > 1
        )
        """
        
        cursor.execute(delete_query)
        connection.commit()
        
        # Check the results
        cursor.execute("SELECT PersonName, COUNT(*) as count FROM tblFaceEmbeddings GROUP BY PersonName")
        new_counts = cursor.fetchall()
        
        print("\nAfter removing duplicates:")
        for name, count in new_counts:
            print(f"{name}: {count} face")
            
        connection.close()
        print("\nDuplicate removal completed successfully!")
        
    except Exception as e:
        logging.error(f"Error removing duplicates: {e}")
        raise

if __name__ == "__main__":
    print("Face Database Duplicate Removal")
    print("==============================")
    delete_duplicates()
    print("\nDone!") 
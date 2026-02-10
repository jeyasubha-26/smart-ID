"""
Attendance Manager for Smart ID - Face Recognition Based Attendance System

This module handles:
1. Creating and managing SQLite database for attendance
2. Inserting attendance records with student name, status, and timestamp
3. Retrieving attendance records and generating reports
4. Simple database operations for attendance tracking
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AttendanceManager:
    """
    Manages attendance data using SQLite database.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the attendance manager.
        
        Args:
            db_path: Path to SQLite database file (optional)
        """
        if db_path is None:
            self.base_dir = Path(__file__).parent.parent
            self.db_dir = self.base_dir / "attendance"
            self.db_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = self.db_dir / "attendance.db"
        else:
            self.db_path = Path(db_path)
        
        # Initialize database
        self._create_database()
        logger.info(f"Attendance manager initialized with database: {self.db_path}")
    
    def _create_database(self):
        """Create database and attendance table if they don't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create attendance table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS attendance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        student_name TEXT NOT NULL,
                        attendance_status TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        date TEXT NOT NULL,
                        session_id TEXT,
                        confidence_score REAL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for better performance
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_student_name ON attendance(student_name)
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_date ON attendance(date)
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_timestamp ON attendance(timestamp)
                ''')
                
                conn.commit()
                logger.info("Database and attendance table created successfully")
                
        except Exception as e:
            logger.error(f"Error creating database: {e}")
            raise
    
    def mark_attendance(self, student_name: str, attendance_status: str = "present", 
                       confidence_score: Optional[float] = None, 
                       session_id: Optional[str] = None) -> bool:
        """
        Mark attendance for a student.
        
        Args:
            student_name: Name of the student
            attendance_status: Attendance status ("present", "absent", "late")
            confidence_score: Face recognition confidence score (optional)
            session_id: Session identifier for grouping (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get current timestamp and date
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
            date = now.strftime("%Y-%m-%d")
            
            # Validate attendance status
            valid_statuses = ["present", "absent", "late"]
            if attendance_status not in valid_statuses:
                logger.warning(f"Invalid attendance status: {attendance_status}")
                attendance_status = "present"
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Insert attendance record
                cursor.execute('''
                    INSERT INTO attendance 
                    (student_name, attendance_status, timestamp, date, session_id, confidence_score)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (student_name, attendance_status, timestamp, date, session_id, confidence_score))
                
                conn.commit()
                logger.info(f"Attendance marked for {student_name}: {attendance_status} at {timestamp}")
                return True
                
        except Exception as e:
            logger.error(f"Error marking attendance for {student_name}: {e}")
            return False
    
    def mark_present(self, student_name: str, confidence_score: Optional[float] = None, 
                    session_id: Optional[str] = None) -> bool:
        """
        Mark student as present.
        
        Args:
            student_name: Name of the student
            confidence_score: Face recognition confidence score
            session_id: Session identifier
            
        Returns:
            True if successful, False otherwise
        """
        return self.mark_attendance(student_name, "present", confidence_score, session_id)
    
    def mark_absent(self, student_name: str, session_id: Optional[str] = None) -> bool:
        """
        Mark student as absent.
        
        Args:
            student_name: Name of the student
            session_id: Session identifier
            
        Returns:
            True if successful, False otherwise
        """
        return self.mark_attendance(student_name, "absent", None, session_id)
    
    def get_attendance_by_date(self, date: str) -> List[Dict]:
        """
        Get attendance records for a specific date.
        
        Args:
            date: Date in YYYY-MM-DD format
            
        Returns:
            List of attendance records as dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM attendance 
                    WHERE date = ? 
                    ORDER BY timestamp
                ''', (date,))
                
                records = [dict(row) for row in cursor.fetchall()]
                logger.info(f"Retrieved {len(records)} attendance records for {date}")
                return records
                
        except Exception as e:
            logger.error(f"Error getting attendance for {date}: {e}")
            return []
    
    def get_attendance_by_student(self, student_name: str, 
                                 start_date: Optional[str] = None, 
                                 end_date: Optional[str] = None) -> List[Dict]:
        """
        Get attendance records for a specific student.
        
        Args:
            student_name: Name of the student
            start_date: Start date in YYYY-MM-DD format (optional)
            end_date: End date in YYYY-MM-DD format (optional)
            
        Returns:
            List of attendance records as dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                if start_date and end_date:
                    cursor.execute('''
                        SELECT * FROM attendance 
                        WHERE student_name = ? AND date BETWEEN ? AND ?
                        ORDER BY timestamp DESC
                    ''', (student_name, start_date, end_date))
                else:
                    cursor.execute('''
                        SELECT * FROM attendance 
                        WHERE student_name = ?
                        ORDER BY timestamp DESC
                    ''', (student_name,))
                
                records = [dict(row) for row in cursor.fetchall()]
                logger.info(f"Retrieved {len(records)} attendance records for {student_name}")
                return records
                
        except Exception as e:
            logger.error(f"Error getting attendance for {student_name}: {e}")
            return []
    
    def get_attendance_summary(self, date: str) -> Dict[str, int]:
        """
        Get attendance summary for a specific date.
        
        Args:
            date: Date in YYYY-MM-DD format
            
        Returns:
            Dictionary with attendance summary statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT attendance_status, COUNT(*) as count
                    FROM attendance 
                    WHERE date = ?
                    GROUP BY attendance_status
                ''', (date,))
                
                results = dict(cursor.fetchall())
                
                summary = {
                    'present': results.get('present', 0),
                    'absent': results.get('absent', 0),
                    'late': results.get('late', 0),
                    'total': sum(results.values())
                }
                
                logger.info(f"Attendance summary for {date}: {summary}")
                return summary
                
        except Exception as e:
            logger.error(f"Error getting attendance summary for {date}: {e}")
            return {'present': 0, 'absent': 0, 'late': 0, 'total': 0}
    
    def get_unique_students(self) -> List[str]:
        """
        Get list of all unique students in the database.
        
        Returns:
            List of unique student names
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT DISTINCT student_name FROM attendance 
                    ORDER BY student_name
                ''')
                
                students = [row[0] for row in cursor.fetchall()]
                logger.info(f"Found {len(students)} unique students")
                return students
                
        except Exception as e:
            logger.error(f"Error getting unique students: {e}")
            return []
    
    def mark_batch_attendance(self, attendance_data: List[Dict], 
                             session_id: Optional[str] = None) -> int:
        """
        Mark attendance for multiple students at once.
        
        Args:
            attendance_data: List of dictionaries with student attendance data
            session_id: Session identifier for grouping
            
        Returns:
            Number of successfully marked attendance records
        """
        success_count = 0
        
        try:
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
            date = now.strftime("%Y-%m-%d")
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for data in attendance_data:
                    try:
                        student_name = data.get('student_name')
                        attendance_status = data.get('attendance_status', 'present')
                        confidence_score = data.get('confidence_score')
                        
                        if not student_name:
                            continue
                        
                        # Validate attendance status
                        valid_statuses = ["present", "absent", "late"]
                        if attendance_status not in valid_statuses:
                            attendance_status = "present"
                        
                        cursor.execute('''
                            INSERT INTO attendance 
                            (student_name, attendance_status, timestamp, date, session_id, confidence_score)
                            VALUES (?, ?, ?, ?, ?, ?)
                        ''', (student_name, attendance_status, timestamp, date, session_id, confidence_score))
                        
                        success_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Error marking attendance for {data.get('student_name', 'unknown')}: {e}")
                        continue
                
                conn.commit()
                logger.info(f"Successfully marked attendance for {success_count} students")
                return success_count
                
        except Exception as e:
            logger.error(f"Error in batch attendance marking: {e}")
            return 0
    
    def clear_old_records(self, days_to_keep: int = 30) -> int:
        """
        Clear old attendance records to manage database size.
        
        Args:
            days_to_keep: Number of days to keep records (default: 30)
            
        Returns:
            Number of deleted records
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    DELETE FROM attendance 
                    WHERE date < date('now', '-{} days')
                '''.format(days_to_keep))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                logger.info(f"Deleted {deleted_count} old attendance records")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Error clearing old records: {e}")
            return 0


def main():
    """
    Main function to test the attendance manager.
    """
    try:
        # Initialize attendance manager
        attendance_mgr = AttendanceManager()
        
        # Test marking attendance
        logger.info("Testing attendance marking...")
        
        # Mark some sample attendance
        attendance_mgr.mark_present("John Doe", confidence_score=0.95, session_id="session_001")
        attendance_mgr.mark_present("Jane Smith", confidence_score=0.87, session_id="session_001")
        attendance_mgr.mark_absent("Bob Johnson", session_id="session_001")
        
        # Get today's date
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Get attendance records
        records = attendance_mgr.get_attendance_by_date(today)
        logger.info(f"Today's attendance records: {len(records)}")
        
        # Get attendance summary
        summary = attendance_mgr.get_attendance_summary(today)
        logger.info(f"Today's attendance summary: {summary}")
        
        # Get unique students
        students = attendance_mgr.get_unique_students()
        logger.info(f"Total unique students: {len(students)}")
        
        logger.info("Attendance manager test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()
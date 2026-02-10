"""
Email Notifier for Smart ID - Face Recognition Based Attendance System

This module handles:
1. Sending email notifications to mentor/advisor
2. Formatting attendance data into readable email content
3. SMTP-based email sending with error handling
4. Simple and reliable email notification system
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmailNotifier:
    """
    Handles email notifications for attendance system.
    """
    
    def __init__(self, smtp_server: str = "smtp.gmail.com", smtp_port: int = 587,
                 sender_email: Optional[str] = None, sender_password: Optional[str] = None):
        """
        Initialize the email notifier.
        
        Args:
            smtp_server: SMTP server address (default: Gmail)
            smtp_port: SMTP server port (default: 587 for TLS)
            sender_email: Email address to send from
            sender_password: Email password or app password
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        
        logger.info(f"Email notifier initialized with server: {smtp_server}")
    
    def _create_email_content(self, attendance_data: Dict) -> str:
        """
        Create formatted email content from attendance data.
        
        Args:
            attendance_data: Dictionary containing attendance information
            
        Returns:
            Formatted email body as string
        """
        try:
            date = attendance_data.get('date', datetime.now().strftime("%Y-%m-%d"))
            session_id = attendance_data.get('session_id', 'Unknown')
            total_students = attendance_data.get('total_students', 0)
            present_students = attendance_data.get('present_students', 0)
            absent_students = attendance_data.get('absent_students', 0)
            attendance_rate = attendance_data.get('attendance_rate', 0)
            absent_list = attendance_data.get('absent_list', [])
            
            # Create email content
            email_body = f"""
SMART ID ATTENDANCE SYSTEM - DAILY REPORT

Date: {date}
Session ID: {session_id}

ATTENDANCE SUMMARY:
------------------
Total Students: {total_students}
Present: {present_students}
Absent: {absent_students}
Attendance Rate: {attendance_rate:.1f}%

ABSENT STUDENTS:
----------------
"""
            
            if absent_list:
                for i, student in enumerate(sorted(absent_list), 1):
                    email_body += f"{i}. {student}\n"
            else:
                email_body += "All students are present today!\n"
            
            email_body += f"""

SYSTEM INFORMATION:
------------------
This is an automated notification from the Smart ID Face Recognition Attendance System.
If you have any questions about this report, please contact the system administrator.

Generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
            
            return email_body
            
        except Exception as e:
            logger.error(f"Error creating email content: {e}")
            return "Error generating email content. Please check the system logs."
    
    def send_attendance_notification(self, recipient_email: str, 
                                   attendance_data: Dict,
                                   subject: Optional[str] = None) -> bool:
        """
        Send attendance notification email to mentor/advisor.
        
        Args:
            recipient_email: Email address of the recipient
            attendance_data: Dictionary containing attendance information
            subject: Custom email subject (optional)
            
        Returns:
            True if email sent successfully, False otherwise
        """
        try:
            # Validate inputs
            if not self.sender_email or not self.sender_password:
                logger.error("Sender email credentials not configured")
                return False
            
            if not recipient_email:
                logger.error("Recipient email not provided")
                return False
            
            # Create email subject
            if not subject:
                date = attendance_data.get('date', datetime.now().strftime("%Y-%m-%d"))
                absent_count = attendance_data.get('absent_students', 0)
                subject = f"Attendance Report - {date} ({absent_count} Absent)"
            
            # Create email content
            email_body = self._create_email_content(attendance_data)
            
            # Create message
            message = MIMEMultipart()
            message["From"] = self.sender_email
            message["To"] = recipient_email
            message["Subject"] = subject
            
            # Attach email body
            message.attach(MIMEText(email_body, "plain"))
            
            # Send email using SMTP
            logger.info(f"Sending email to {recipient_email}...")
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                # Start TLS encryption
                server.starttls()
                
                # Login to email account
                server.login(self.sender_email, self.sender_password)
                
                # Send email
                text = message.as_string()
                server.sendmail(self.sender_email, recipient_email, text)
            
            logger.info(f"Email sent successfully to {recipient_email}")
            return True
            
        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"SMTP authentication failed: {e}")
            logger.error("Please check your email credentials or use an app password")
            return False
            
        except smtplib.SMTPRecipientsRefused as e:
            logger.error(f"Recipient email refused: {e}")
            return False
            
        except smtplib.SMTPServerDisconnected as e:
            logger.error(f"SMTP server disconnected: {e}")
            return False
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False
    
    def send_multiple_notifications(self, recipient_emails: List[str], 
                                  attendance_data: Dict,
                                  subject: Optional[str] = None) -> Dict[str, bool]:
        """
        Send attendance notifications to multiple recipients.
        
        Args:
            recipient_emails: List of recipient email addresses
            attendance_data: Dictionary containing attendance information
            subject: Custom email subject (optional)
            
        Returns:
            Dictionary mapping email addresses to send status
        """
        results = {}
        
        for email in recipient_emails:
            if email:  # Skip empty email addresses
                results[email] = self.send_attendance_notification(email, attendance_data, subject)
            else:
                results[email] = False
        
        success_count = sum(1 for success in results.values() if success)
        logger.info(f"Sent notifications to {success_count}/{len(recipient_emails)} recipients")
        
        return results
    
    def test_email_connection(self) -> bool:
        """
        Test connection to SMTP server.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if not self.sender_email or not self.sender_password:
                logger.error("Email credentials not configured")
                return False
            
            logger.info("Testing SMTP connection...")
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                # Start TLS encryption
                server.starttls()
                
                # Test login
                server.login(self.sender_email, self.sender_password)
            
            logger.info("SMTP connection test successful")
            return True
            
        except Exception as e:
            logger.error(f"SMTP connection test failed: {e}")
            return False
    
    def configure_credentials(self, sender_email: str, sender_password: str):
        """
        Configure email sender credentials.
        
        Args:
            sender_email: Email address to send from
            sender_password: Email password or app password
        """
        self.sender_email = sender_email
        self.sender_password = sender_password
        logger.info("Email credentials configured")


def main():
    """
    Main function to test the email notifier.
    """
    try:
        # Initialize email notifier
        # Note: Replace with actual email credentials for testing
        notifier = EmailNotifier(
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            sender_email="your_email@gmail.com",  # Replace with actual email
            sender_password="your_app_password"    # Replace with actual app password
        )
        
        # Test email connection
        if not notifier.test_email_connection():
            logger.error("Email connection test failed. Please check credentials.")
            return
        
        # Sample attendance data for testing
        sample_attendance_data = {
            'date': datetime.now().strftime("%Y-%m-%d"),
            'session_id': f"test_session_{datetime.now().strftime('%H%M%S')}",
            'total_students': 25,
            'present_students': 22,
            'absent_students': 3,
            'attendance_rate': 88.0,
            'absent_list': ['John Doe', 'Jane Smith', 'Bob Johnson'],
            'present_details': {}  # Not used in email
        }
        
        # Test email sending (replace with actual recipient)
        recipient_email = "mentor@university.edu"  # Replace with actual email
        success = notifier.send_attendance_notification(recipient_email, sample_attendance_data)
        
        if success:
            logger.info("Test email sent successfully!")
        else:
            logger.error("Failed to send test email")
        
        logger.info("Email notifier test completed!")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()
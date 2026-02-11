# Smart ID – Face Recognition Based Attendance System

## Abstract

Smart ID is an AI-powered attendance management system that uses face recognition to automatically identify individuals and mark attendance. The system compares facial images from a test dataset against a registered reference dataset using DeepFace (FaceNet model). When a match is found, the individual is marked present; otherwise, the face is labeled as unknown. Attendance records are stored in a structured SQLite database with date and time information. The system also generates and sends automated email notifications summarizing attendance results. A modular architecture is followed, separating dataset handling, face recognition, attendance management, and email notification components. The current implementation works using static datasets without requiring live webcam input. This ensures controlled testing and privacy compliance. The project demonstrates practical applications of computer vision, identity verification, and automation in academic environments. Smart ID can be extended for real-world institutional deployment with live camera integration and centralized database connectivity.

---

## Output

### Console Output Screenshot

![Console Output](console_output.png)

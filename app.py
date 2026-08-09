"""
Top-level application launcher.
Run with:
    python app.py
Open browser at:
    http://127.0.0.1:8000
"""
import uvicorn

if __name__ == "__main__":
    print("\n========================================================================")
    print(" Starting Tree Species Identification Local Application Server...")
    print(" Web Browser URL: http://127.0.0.1:8000")
    print("========================================================================\n")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)

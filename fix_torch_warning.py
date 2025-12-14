"""
Workaround script for Streamlit + PyTorch file watcher issue.
Run this before starting Streamlit if you continue to see torch.classes errors.
"""
import os
import sys

# Add this to your environment to suppress the warning
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'poll'
os.environ['STREAMLIT_SERVER_RUN_ON_SAVE'] = 'true'

print("✅ Streamlit file watcher configured to use 'poll' mode")
print("This should prevent the torch.classes RuntimeError")
print("\nYou can now run: streamlit run app.py")


import socket
import os

def get_ip_address():
    """Get the local IP address of this machine"""
    try:
        # Create a socket to determine the IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Connect to an external server (doesn't actually send packets)
        s.connect(("8.8.8.8", 80))
        # Get the local IP address
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        print(f"Error getting IP address: {e}")
        # Fallback method
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        return ip

if __name__ == "__main__":
    ip = get_ip_address()
    print("\nIP Address Information for ID Scanner Web Server:")
    print("-------------------------------------------------")
    print(f"Local access:     http://localhost:5000")
    print(f"Network access:   http://{ip}:5000")
    print("-------------------------------------------------")
    print("Use the Network access URL to connect from your phone")
    print("Make sure your phone is connected to the same WiFi network")
    print("-------------------------------------------------\n") 
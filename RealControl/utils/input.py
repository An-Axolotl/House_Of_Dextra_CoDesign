"""
Cross-platform keyboard input utilities.
"""
import os
import sys
import time


def setup_input():
    """Set up platform-specific input handling."""
    if os.name != 'nt':
        import tty
        import termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        return fd, old_settings
    return None, None


def getch(block=False, timeout=None, flush=False, fd=None, old_settings=None):
    """
    Cross-platform character input function.
    
    Args:
        block: If True and timeout is None, blocks until key is pressed
        timeout: Maximum time to wait for input (seconds)
        flush: If True, flush input buffer before reading
        fd: File descriptor for Unix systems
        old_settings: Terminal settings for Unix systems
        
    Returns:
        str or None: Character pressed, or None if no input
    """
    if os.name == 'nt':
        import msvcrt
        
        # Blocking: wait for a key
        if block and timeout is None:
            return msvcrt.getch().decode(errors="ignore")
        
        # Timed or non-blocking
        if timeout is not None:
            # Emulate a timed wait
            start = time.time()
            while time.time() - start < timeout:
                if msvcrt.kbhit():
                    return msvcrt.getch().decode(errors="ignore")
                time.sleep(0.01)
            return None
        
        return msvcrt.getch().decode(errors="ignore") if msvcrt.kbhit() else None
    
    else:  # Unix-like systems
        import tty
        import termios
        import select
        
        if fd is None:
            fd = sys.stdin.fileno()
        if old_settings is None:
            old_settings = termios.tcgetattr(fd)
        
        old = termios.tcgetattr(fd)
        try:
            if flush:
                termios.tcflush(fd, termios.TCIFLUSH)
            tty.setcbreak(fd)  # raw-ish mode, no enter needed
            sys.stdout.flush()
            
            if block and timeout is None:
                # Truly block for one char
                return sys.stdin.read(1)
            
            # Timed or non-blocking
            wait = 0 if timeout is None else timeout
            r, _, _ = select.select([fd], [], [], wait)
            return sys.stdin.read(1) if r else None
            
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
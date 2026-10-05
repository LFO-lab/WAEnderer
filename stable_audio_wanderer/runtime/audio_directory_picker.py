"""Native directory dialog, run as a subprocess to keep Tk on its main thread."""
import json
import os
import sys


def main():
    root = None
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        initial = sys.argv[1] if len(sys.argv) > 1 else ""
        options = {"title": "Choose audio source directory", "mustexist": True}
        if os.path.isdir(initial):
            options["initialdir"] = initial
        # An explicit parent makes macOS use a sheet attached to the withdrawn
        # root. Its position can be off-screen and the sheet cannot be moved.
        # Omit the parent so the native picker is a standalone dialog.
        path = filedialog.askdirectory(**options)
        result = {"audio_dir": path, "cancelled": not bool(path)}
    except Exception as exc:
        result = {"error": "Folder picker requires Python Tk support and a desktop display. "
                           + str(exc)}
    finally:
        if root is not None:
            root.destroy()
    print(json.dumps(result))


if __name__ == "__main__":
    main()

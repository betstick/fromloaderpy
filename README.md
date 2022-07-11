# Python extension for Fromloader
Gives a barebones interface for Fromloader so you can read files in Python very fast.

# How to compile
CD to this dir, run this: `python build_ext.py install`
<br/>
The python module with automatically get put into a dir above this called `fromloaderpy_d`.
<br/>
Should run on Windows and Linux. Debugging on Windows is so borked I had to make this repo right now just to compile and debug on Linux. If anybody knows how to debug a CMake C++/Python extension in VS/VSCode, let me know. I can't figure it out, and I already like GDB.
<br/>
Don't try to compile with VS CMake GUI. It does weird things and there's no point in using it. Just use the CLI.

# Notes
Never make Python extensions. Its terrible.

# License
GPLv3